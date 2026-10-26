"""Layer-streaming inference: run models larger than safe GPU memory.

A model is loaded lazily, then split by a byte budget into resident layers
(materialized once, kept) and streamed layers. A streamed layer's weights are
read from its safetensors file when the layer runs, evaluated, and dropped
again, so every forward pass re-reads them. A background thread reads the
next streamed layers ahead into the OS page cache: page-cache memory is clean
and evictable, so when RAM is free a pass runs at memcpy speed, and under
pressure macOS reclaims it instead of starving WindowServer (the 3-bit 27B
fully resident on 16GB ended in a watchdog panic).

One pass costs one read of the streamed bytes regardless of how many tokens or
sequences it carries, so streamed models want batched generation (eval) and
speculative decoding (chat). Speculative decoding needs rewindable caches;
hybrid models' linear-attention state is recurrent, so RewindableArraysCache
keeps the state after each recent token and linear layers step through short
inputs one token at a time to produce those snapshots.
"""
from __future__ import annotations

import json
import os
import struct
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.cache import ArraysCache

from .registry import machine_ram_gb

# Resident-weight budget when none is given: at most this fraction of RAM, and
# no more than what is available right now minus a margin, leaving the rest to
# macOS, open apps, the KV cache, and the page cache that streaming feeds on.
DEFAULT_BUDGET_FRACTION = 0.4
AVAILABLE_MARGIN_GB = 1.5
MLX_CACHE_LIMIT = 1 << 30
ADAPTER_LEAVES = {"lora_a", "lora_b", "m"}
ADAPTER_CONFIG = "adapter_config.json"
_READ_CHUNK = 64 << 20


@dataclass(frozen=True)
class _Span:
    file: str
    start: int
    end: int


def available_gb() -> float:
    """Free + inactive + speculative pages (what macOS can hand out without
    compressing or swapping), from vm_stat."""
    import re
    import subprocess
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1))
    pages = sum(int(m.group(1)) for m in re.finditer(
        r"Pages (?:free|inactive|speculative):\s+(\d+)", out))
    return pages * page / 2**30


def sequence_state_gb(model_dir: str, tokens: int = 512) -> float:
    """Per-sequence cache estimate: fp32 recurrent state of each linear-attention
    layer (fixed size) plus bf16 K/V of each full-attention layer at `tokens`.
    Batched generation holds one per sequence, which a batch of 16 on a 27B
    hybrid (~150MB each) turns into gigabytes."""
    cfg = json.loads((Path(resolve_model_dir(model_dir)[0]) / "config.json").read_text())
    t = cfg.get("text_config", cfg)
    n = t["num_hidden_layers"]
    types = t.get("layer_types") or ["full_attention"] * n
    n_linear = sum(x == "linear_attention" for x in types)
    head_dim = t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]
    kv = 2 * t.get("num_key_value_heads", t["num_attention_heads"]) * head_dim * 2 * tokens
    lin = (t.get("linear_num_value_heads", 0) * t.get("linear_value_head_dim", 0)
           * t.get("linear_key_head_dim", 0) * 4)
    return ((n - n_linear) * kv + n_linear * lin) / 2**30


def default_budget_gb() -> float:
    return max(1.0, min(machine_ram_gb() * DEFAULT_BUDGET_FRACTION,
                        available_gb() - AVAILABLE_MARGIN_GB))


def scan_weights(model_dir: str) -> dict[str, _Span]:
    """key -> absolute byte span, from every model*.safetensors header."""
    spans: dict[str, _Span] = {}
    for f in sorted(Path(model_dir).glob("model*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        base = 8 + n
        for key, meta in header.items():
            if key != "__metadata__":
                lo, hi = meta["data_offsets"]
                spans[key] = _Span(str(f), base + lo, base + hi)
    return spans


class _Prefetcher:
    """Reads byte spans in a background thread to warm the OS page cache."""

    def __init__(self):
        self._queue: deque[tuple[int, list[_Span]]] = deque()
        self._pending: set[int] = set()
        self._cv = threading.Condition()
        self._fds: dict[str, int] = {}
        self._buf = memoryview(bytearray(_READ_CHUNK))
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, tag: int, spans: list[_Span]) -> None:
        with self._cv:
            if tag not in self._pending:
                self._pending.add(tag)
                self._queue.append((tag, spans))
                self._cv.notify()

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._queue:
                    self._cv.wait()
                tag, spans = self._queue.popleft()
            for s in spans:
                fd = self._fds.get(s.file)
                if fd is None:
                    fd = self._fds[s.file] = os.open(s.file, os.O_RDONLY)
                off = s.start
                while off < s.end:
                    n = os.preadv(fd, [self._buf[: min(_READ_CHUNK, s.end - off)]], off)
                    if n <= 0:
                        break
                    off += n
            with self._cv:
                self._pending.discard(tag)


class RewindableArraysCache(ArraysCache):
    """ArraysCache that can be trimmed by restoring per-token state snapshots.

    `history[-1]` is the current state; trim(n) restores the state from n
    tokens ago. Only the last `max_history` token states are kept, and a bulk
    call (prefill) resets the history to its final state.
    """

    def __init__(self, size, max_history: int = 8, left_padding=None):
        super().__init__(size, left_padding=left_padding)
        self.max_history = max_history
        self.history: list[list] = []

    def record(self) -> None:
        self.history.append(list(self.cache))
        if len(self.history) > self.max_history:
            del self.history[0]

    def is_trimmable(self):
        return True

    def trim(self, n):
        if n <= 0:
            return 0
        if n >= len(self.history):
            raise RuntimeError(
                f"cannot rewind {n} tokens: only {len(self.history) - 1} snapshots kept")
        del self.history[-n:]
        self.cache = list(self.history[-1])
        return n


def make_rewindable_cache(model, max_history: int = 8) -> list:
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    return [RewindableArraysCache(len(c.cache), max_history)
            if type(c) is ArraysCache else c for c in cache]


class EngineLayer(nn.Module):
    """Wraps one decoder layer: streams its weights and/or steps it per token.

    Pass-through for resident layers without a rewindable cache, so the lazy
    graph across resident layers is untouched.
    """

    def __init__(self, layer: nn.Module, engine: "StreamEngine | None",
                 index: int, streamed: bool):
        super().__init__()
        self.layer = layer
        self._engine = engine
        self._index = index
        self._streamed = streamed
        self.set_ablation(None)

    def set_ablation(self, dirs: mx.array | None) -> None:
        """Remove orthonormal rows `dirs` [k, D] from this layer's output (and
        layer 0's input, i.e. the embedding): with every layer set, identical
        to orthogonalizing all residual writers. Kept outside the parameter
        tree (object attribute), so it never counts as a weight."""
        object.__setattr__(self, "_ablate", dirs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            inner = dict.get(self, "layer")
            if inner is None:
                raise
            return getattr(inner, name)

    def __call__(self, x, mask=None, cache=None):
        if self._streamed:
            self._engine.prefetch_after(self._index)
        if self._ablate is not None and self._index == 0:
            x = self._project(x)
        if isinstance(cache, RewindableArraysCache):
            out = self._step(x, mask, cache)
        else:
            out = self.layer(x, mask=mask, cache=cache)
        if self._ablate is not None:
            out = self._project(out)
        if self._streamed:
            state = [a for _, a in tree_flatten(cache.state)] if cache is not None else []
            mx.eval(out, [a for a in state if isinstance(a, mx.array)])
            self._engine.evict(self._index)
        return out

    def _project(self, h):
        # float32: bf16 rounding of large residual norms compounds over layers
        d = self._ablate.astype(mx.float32)
        return h - ((h.astype(mx.float32) @ d.T) @ d).astype(h.dtype)

    def _step(self, x, mask, cache):
        S = x.shape[1]
        if S == 1 or S >= cache.max_history:
            out = self.layer(x, mask=mask, cache=cache)
            if S > 1:
                cache.history.clear()
            cache.record()
            return out
        outs = []
        for t in range(S):
            m = mask[:, t : t + 1] if isinstance(mask, mx.array) else None
            outs.append(self.layer(x[:, t : t + 1], mask=m, cache=cache))
            cache.record()
        return mx.concatenate(outs, axis=1)


def _layers_path(model) -> tuple[list, str]:
    """The decoder-layer list object itself (`model.layers` can be a slice
    copy, e.g. Qwen3.5's pipeline_layers) and its parameter-name prefix."""
    first = model.layers[0]
    for name, mod in model.named_modules():
        if mod is first:
            *parents, attr, _ = name.split(".")
            owner = model
            for part in parents:
                owner = owner[part]
            return owner[attr], ".".join([*parents, attr]) + "."
    raise RuntimeError("could not locate decoder layers in module tree")


class StreamEngine:
    def __init__(self, model, model_dir: str, budget_gb: float | None):
        self.model = model
        self.layers, prefix = _layers_path(model)
        spans = scan_weights(model_dir)
        n = len(self.layers)
        self._names: list[dict[str, str]] = []   # per layer: relative -> file key
        self._spans: list[list[_Span]] = []
        layer_bytes = []
        for i, layer in enumerate(self.layers):
            keys = {}
            for rel, _ in tree_flatten(layer.parameters()):
                if rel.rsplit(".", 1)[-1] in ADAPTER_LEAVES:
                    continue  # adapter params: in memory, tiny, stay resident
                # LoRA wrapping (abliteration adapters) nests the base as .linear
                key = prefix + f"{i}." + rel.replace(".linear.", ".")
                if key not in spans:
                    raise RuntimeError(
                        f"layer {i}: {rel} not found verbatim in safetensors; "
                        "streaming needs an MLX-converted model")
                keys[rel] = key
            self._names.append(keys)
            self._spans.append([spans[v] for v in keys.values()])
            layer_bytes.append(sum(s.end - s.start for s in self._spans[-1]))
        other_bytes = sum(s.end - s.start for k, s in spans.items()
                          if not k.startswith(prefix))

        budget = (budget_gb if budget_gb is not None else default_budget_gb()) * 2**30
        n_resident, used = 0, other_bytes
        while n_resident < n and used + layer_bytes[n_resident] <= budget:
            used += layer_bytes[n_resident]
            n_resident += 1
        # spread streamed layers evenly so each read overlaps resident compute
        n_streamed = n - n_resident
        self.streamed = sorted({round((j + 0.5) * n / n_streamed - 0.5)
                                for j in range(n_streamed)}) if n_streamed else []
        streamed_set = set(self.streamed)
        self._next = {s: self.streamed[(j + 1) % len(self.streamed)]
                      for j, s in enumerate(self.streamed)}
        self.resident_gb = (other_bytes + sum(
            b for i, b in enumerate(layer_bytes) if i not in streamed_set)) / 2**30
        self.streamed_gb = sum(layer_bytes[i] for i in self.streamed) / 2**30
        self._prefetcher = _Prefetcher() if self.streamed else None

        streamed_names = {f"{prefix}{i}.{rel}" for i in streamed_set for rel in self._names[i]}
        resident = [a for k, a in tree_flatten(model.parameters()) if k not in streamed_names]
        for i in range(n):
            self.layers[i] = EngineLayer(self.layers[i], self, i, i in streamed_set)
        for j in range(0, len(resident), 64):
            mx.eval(resident[j : j + 64])
        mx.clear_cache()
        for i in self.streamed:
            self.evict(i)
        mx.set_cache_limit(MLX_CACHE_LIMIT)
        self._patch_wired_limit()

    def evict(self, i: int) -> None:
        """Swap layer i's (materialized) weights for fresh lazy file loads."""
        loaded: dict[str, dict] = {}
        weights = []
        for (rel, key), span in zip(self._names[i].items(), self._spans[i]):
            if span.file not in loaded:
                loaded[span.file] = mx.load(span.file)  # lazy: header only
            weights.append((rel, loaded[span.file][key]))
        self.layers[i].layer.load_weights(weights, strict=False)

    def prefetch_after(self, i: int) -> None:
        j = self._next[i]
        self._prefetcher.submit(j, self._spans[j])

    def _patch_wired_limit(self) -> None:
        cap_wired_limit(self.resident_gb + 1.0)

    def describe(self) -> str:
        n = len(self.layers)
        return (f"{n - len(self.streamed)}/{n} layers resident "
                f"({self.resident_gb:.1f}GB), {len(self.streamed)} streamed "
                f"({self.streamed_gb:.1f}GB read per forward pass)")


def cap_wired_limit(gb: float) -> None:
    """mlx_lm wires up to the device's recommended working set (~10.7GB on
    16GB) while generating; wired buffers can't be paged, which is how a 16GB
    Mac starves WindowServer. Wire at most `gb` instead.

    Patched through sys.modules: `import mlx_lm.generate` binds the
    `generate` function that mlx_lm/__init__.py re-exports, not the module,
    so patching that name silently does nothing."""
    import importlib
    import sys
    limit = int(gb * 2**30)

    def wired() -> int:
        return mx.set_wired_limit(limit)
    for name in ("mlx_lm.utils", "mlx_lm.generate", "mlx_lm.server"):
        mod = sys.modules.get(name) or importlib.import_module(name)
        if hasattr(mod, "maybe_set_recommended_wired_limit"):
            mod.maybe_set_recommended_wired_limit = wired


def resolve_model_dir(model_dir: str) -> tuple[str, str | None]:
    """(weights dir, adapter dir or None). An abliteration dir for a quantized
    model holds only an adapter whose config names its base model."""
    cfg = Path(model_dir) / ADAPTER_CONFIG
    if cfg.exists():
        return json.loads(cfg.read_text())["base_model"], model_dir
    return model_dir, None


FINGERPRINT_SUFFIXES = (".safetensors", ".json", ".jinja", ".model", ".txt")


def files_fingerprint(*paths: str) -> str:
    """Cheap identity of files or model dirs: name, size and mtime of each
    weight, config, tokenizer and chat-template file. An adapter dir also
    covers its base model."""
    import hashlib
    h = hashlib.sha256()
    for path in paths:
        p = Path(path)
        if p.is_dir():
            base, adapter = resolve_model_dir(path)
            files = [f for d in filter(None, (base, adapter)) for f in sorted(Path(d).iterdir())
                     if f.is_file() and f.suffix in FINGERPRINT_SUFFIXES]
        else:
            files = [p]
        for f in files:
            st = f.stat()
            h.update(f"{f}:{st.st_size}:{st.st_mtime_ns}\n".encode())
    return h.hexdigest()[:16]


def load_streamed(model_dir: str, budget_gb: float | None = None, prepare=None):
    """(model, tokenizer, engine) with at most ~budget_gb of weights resident.
    `prepare(model)` runs before streaming is set up, e.g. to wrap modules
    (a wrapper must hold the original as `.linear` to keep weight names)."""
    from mlx_lm import load
    base, adapter = resolve_model_dir(model_dir)
    model, tokenizer = load(base, adapter_path=adapter, lazy=True)
    if prepare is not None:
        prepare(model)
    engine = StreamEngine(model, base, budget_gb)
    return model, tokenizer, engine


def load_for_inference(model_dir: str, budget_gb: float | None = None,
                       stream: bool | None = None):
    """(model, tokenizer, engine-or-None). stream=None streams only when the
    weights exceed the resident budget; stream=False loads fully (guarded)."""
    from mlx_lm import load

    from .registry import check_inference_fits, weights_gb
    base, adapter = resolve_model_dir(model_dir)
    if stream is None:
        budget = budget_gb if budget_gb is not None else default_budget_gb()
        stream = weights_gb(base) > budget
    if stream:
        return load_streamed(model_dir, budget_gb)
    check_inference_fits(base, force=True)  # explicit or fits: warn only
    model, tokenizer = load(base, adapter_path=adapter)
    return model, tokenizer, None


def wrap_for_rewind(model) -> None:
    """Make a fully resident model's layers rewind-capable (speculative draft)."""
    layers, _ = _layers_path(model)
    for i in range(len(layers)):
        if not isinstance(layers[i], EngineLayer):
            layers[i] = EngineLayer(layers[i], None, i, False)
