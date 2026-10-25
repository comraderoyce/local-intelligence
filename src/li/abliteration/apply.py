"""Apply abliteration by orthogonalizing refusal directions out of the weights.

Directions D [k, D] (orthonormal) come from `measure`; every residual-output
projection (self_attn.o_proj, linear_attn.out_proj, mlp.down_proj, MoE
expert/shared down_proj) of layers >= `from_layer` becomes
W' = (I - alpha D^T D) W, and with `embed` the token embedding becomes
E' = E (I - alpha D^T D), so nothing writes the directions into the residual
stream.

Quantized models get a rank-k adapter instead of edited weights: the source
weights sit exactly on their quantization grid, so requantizing an edit at the
same bits snaps ~90% of it back (measured on Qwen3.8-27B 3-bit: 91% of the
refusal component survived). The adapter applies the same correction exactly
as mlx_lm LoRA on the untouched base (lora_a = W^T D^T, lora_b = -alpha D).

Unquantized weights are edited in place in float32. Shards are loaded lazily
and rewritten in ~1GB parts (index updated to match), so peak memory is one
part, not one multi-GB source shard.
"""
from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors.numpy import safe_open as np_safe_open

# Shard IO uses mx.load/mx.save_safetensors: mlx-community shards carry bf16
# scales/biases, which the safetensors numpy and mlx readers in this env
# cannot materialize (TypeError: data type 'bfloat16' not understood).

PART_BYTES = 1 << 30

# `language_model.` prefix: multimodal-wrapped checkpoints (Qwen3.5+);
# `linear_attn.out_proj`: Gated DeltaNet layers in hybrid models
TARGET_RE = re.compile(
    r"^(?:language_model\.)?model\.layers\.(\d+)\."
    r"(self_attn\.o_proj|linear_attn\.out_proj|mlp\.down_proj|"
    r"mlp\.experts\.\d+\.down_proj|mlp\.shared_expert\.down_proj)\.weight$"
)


EMBED_RE = re.compile(r"^(?:language_model\.)?model\.embed_tokens\.weight$")

# directions this close to the span of those already chosen add nothing
MIN_RESIDUAL = 0.5


def select_directions(stats_path: str, layer: int, pos: int | None = None,
                      n_dirs: int = 1, project_harmless: bool = False
                      ) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Orthonormal refusal directions [k, D] and the (layer, pos) of each.

    The first is measured at (`layer`, `pos`); pos is a negative offset into
    the template suffix, default the best-scoring position at that layer.
    Further directions are the next best-scoring (layer, pos) in the upper 2/3
    of the network that are not already (nearly) spanned. project_harmless
    first removes each direction's component along its harmless mean, so the
    edit leaves what harmless prompts write at that point alone.
    """
    with np_safe_open(stats_path, framework="numpy") as f:
        keys = set(f.keys())
        if "directions_pos" in keys:
            dirs, scores, ben = (f.get_tensor(k).astype(np.float32) for k in
                                 ("directions_pos", "scores_pos", "harmless_mean_pos"))
        else:  # last-position-only stats from before per-position measurement
            dirs, scores, ben = (f.get_tensor(k).astype(np.float32)[:, None] for k in
                                 ("directions", "scores", "harmless_mean"))
    n_layers, n_pos, _ = dirs.shape
    idx = int(np.argmax(scores[layer])) if pos is None else pos + n_pos
    if not 0 <= idx < n_pos:
        raise ValueError(f"pos {pos} outside the {n_pos}-token template suffix")

    def candidate(l: int, i: int) -> np.ndarray:
        d = dirs[l, i]
        if project_harmless:
            m = ben[l, i] / np.linalg.norm(ben[l, i])
            d = d - (d @ m) * m
        return d / np.linalg.norm(d)

    basis, used = [candidate(layer, idx)], [(layer, idx - n_pos)]
    order = np.argsort(-scores[n_layers // 3:], axis=None)
    for flat in order:
        if len(basis) >= n_dirs:
            break
        l, i = n_layers // 3 + flat // n_pos, flat % n_pos
        v = candidate(l, i)
        for b in basis:
            v = v - (v @ b) * b
        if np.linalg.norm(v) >= MIN_RESIDUAL:
            basis.append(v / np.linalg.norm(v))
            used.append((int(l), int(i) - n_pos))
    return np.stack(basis).astype(np.float32), used


def _quant_params(model_dir: str) -> dict | None:
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    return cfg.get("quantization") or cfg.get("quantization_config")


def _project_out(w: mx.array, dirs: mx.array, alpha: float) -> mx.array:
    """W' = W - alpha * D^T D W for W [out, in], orthonormal rows D [k, out]."""
    assert w.shape[0] == dirs.shape[-1], (
        f"direction dim {dirs.shape[-1]} != weight rows {w.shape[0]}")
    orig_dtype = w.dtype
    assert str(orig_dtype) not in ("uint32", "uint16", "uint8", "int32"), (
        f"{orig_dtype} looks like packed quantized codes, not raw weights")
    w32 = w.astype(mx.float32)
    return (w32 - alpha * dirs.T @ (dirs @ w32)).astype(orig_dtype)


def _project_out_rows(e: mx.array, dirs: mx.array, alpha: float) -> mx.array:
    """Embedding rows [vocab, D]: E' = E - alpha * (E D^T) D."""
    e32 = e.astype(mx.float32)
    return (e32 - alpha * (e32 @ dirs.T) @ dirs).astype(e.dtype)


def _modify_quantized(tensors: dict, key: str, param: str,
                      dirs: mx.array, alpha: float, quant: dict) -> None:
    spec = {**quant, **quant.get(param, {})}
    bits, group_size = int(spec["bits"]), int(spec["group_size"])
    # shard layout: <param>.weight is packed, alongside <param>.scales/.biases
    scales, biases = tensors[param + ".scales"], tensors[param + ".biases"]
    w = mx.dequantize(tensors[key], scales, biases,
                      group_size=group_size, bits=bits, dtype=mx.float32)
    w_new = _project_out(w, dirs, alpha)
    wq, s, b = mx.quantize(w_new, group_size=group_size, bits=bits)
    tensors[key] = wq
    tensors[param + ".scales"] = s.astype(scales.dtype)
    tensors[param + ".biases"] = b.astype(biases.dtype)


def _modify_raw(tensors: dict, key: str, dirs: mx.array, alpha: float) -> None:
    if EMBED_RE.match(key):
        tensors[key] = _project_out_rows(tensors[key], dirs, alpha)
    else:
        tensors[key] = _project_out(tensors[key], dirs, alpha)


def _dequantized(tensors: dict, key: str, quant: dict, rows: slice = slice(None)):
    param = key[: -len(".weight")]
    spec = {**quant, **quant.get(param, {})}
    return mx.dequantize(tensors[key][rows], tensors[param + ".scales"][rows],
                         tensors[param + ".biases"][rows], group_size=int(spec["group_size"]),
                         bits=int(spec["bits"]), dtype=mx.float32)


EMBED_CHUNK = 8192

# alpha_of(layer, kind) -> removal strength; layer None for the embedding,
# kind "attn" (o_proj / out_proj), "mlp" (down_proj) or "embed"; 0 = untouched
AlphaFn = Callable[[int | None, str], float]


def _component(key: str) -> tuple[int | None, str] | None:
    if EMBED_RE.match(key):
        return None, "embed"
    m = TARGET_RE.match(key)
    if not m:
        return None
    return int(m.group(1)), ("mlp" if m.group(2).endswith("down_proj") else "attn")


def uniform_alpha(alpha: float, from_layer: int, embed: bool) -> AlphaFn:
    def alpha_of(layer: int | None, kind: str) -> float:
        if kind == "embed":
            return alpha if embed else 0.0
        return alpha if layer >= from_layer else 0.0
    return alpha_of


ADAPTER_KEYS = ("self_attn.o_proj", "linear_attn.out_proj", "mlp.down_proj",
                "mlp.shared_expert.down_proj")


def _write_adapter(model_dir: str, dirs: mx.array, meta: dict, alpha_of: AlphaFn,
                   out_dir: str, quant: dict) -> dict:
    """Rank-k LoRA equal to W' = (I - a D^T D) W on every target, and to
    E' = E (I - a D^T D) on the embedding, a = alpha_of(layer, kind):
    lora_a = W^T D^T (or E D^T), lora_b = -a D, scale 1."""
    src, dst = Path(model_dir), Path(out_dir)
    cfg = json.loads((src / "config.json").read_text())
    n_layers = cfg.get("text_config", cfg)["num_hidden_layers"]
    adapters: dict[str, mx.array] = {}
    keys = list(ADAPTER_KEYS)
    lowest = n_layers
    for shard in sorted(src.glob("*.safetensors")):
        tensors = mx.load(str(shard))
        for key in tensors:
            comp = _component(key)
            if comp is None or (a := alpha_of(*comp)) == 0:
                continue
            layer, kind = comp
            param = key[: -len(".weight")]
            if kind == "embed":
                vocab = tensors[key].shape[0]
                parts = []
                for r in range(0, vocab, EMBED_CHUNK):
                    parts.append(_dequantized(tensors, key, quant, slice(r, r + EMBED_CHUNK))
                                 @ dirs.T)
                    mx.eval(parts[-1])
                adapters[param + ".lora_a"] = mx.concatenate(parts)  # [vocab, k]
                keys.append(param)  # top-level module path, matched model-wide
            else:
                if ".experts." in key:
                    raise NotImplementedError("adapter abliteration of per-expert weights")
                adapters[param + ".lora_a"] = (dirs @ _dequantized(tensors, key, quant)).T
                lowest = min(lowest, layer)
            adapters[param + ".lora_b"] = -a * dirs                  # [in|vocab, k], [k, out]
            mx.eval(adapters[param + ".lora_a"])
        del tensors
        mx.clear_cache()
    if dst.exists():
        for f in dst.iterdir():  # replace any earlier (weight-copy) output
            if f.is_file():
                f.unlink()
    dst.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(dst / "adapters.safetensors"), adapters)
    (dst / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": "lora",
        # LoRA-wraps the last num_layers layers; wrapped modules without saved
        # weights keep lora_b = 0 and are unchanged
        "num_layers": n_layers - lowest,
        "lora_parameters": {"rank": int(dirs.shape[0]), "scale": 1.0, "dropout": 0.0,
                            "keys": keys},
        "base_model": model_dir,
        "abliteration": meta,
    }, indent=2) + "\n")
    return {"modified_tensors": len(adapters) // 2, **meta, "out_dir": out_dir,
            "quantized": True, "adapter": True}


def apply_abliteration(model_dir: str, stats_path: str, layer: int,
                       alpha: float, out_dir: str, from_layer: int | None = None,
                       pos: int | None = None, n_dirs: int = 1, embed: bool = True,
                       project_harmless: bool = False) -> dict:
    # directions measured at (`layer`, `pos`) [+ next best], orthogonalized out
    # of every residual-output projection from `from_layer` upward and out of
    # the embedding (from_layer 0 + embed = canonical Arditi et al.)
    if from_layer is None:
        from_layer = layer
    dirs_np, used = select_directions(stats_path, layer, pos, n_dirs, project_harmless)
    meta = {"layer": layer, "directions_at": used, "project_harmless": project_harmless,
            "from_layer": from_layer, "alpha": alpha, "embed": embed}
    return write_abliteration(model_dir, dirs_np, out_dir,
                              uniform_alpha(alpha, from_layer, embed), meta)


def write_abliteration(model_dir: str, dirs_np: np.ndarray, out_dir: str,
                       alpha_of: AlphaFn, meta: dict) -> dict:
    """Orthogonalize orthonormal `dirs_np` [k, D] out of every target with
    alpha_of(layer, kind) != 0: an adapter for quantized models, an edited
    copy otherwise."""
    src = Path(model_dir)
    dst = Path(out_dir)
    if dst.exists() and dst.resolve() == src.resolve():
        raise ValueError("out_dir must differ from model_dir")
    dst_existed = dst.exists()
    dst.mkdir(parents=True, exist_ok=True)

    quant = _quant_params(model_dir)
    dirs = mx.array(dirs_np)
    if quant:
        result = _write_adapter(model_dir, dirs, meta, alpha_of, out_dir, quant)
        (dst / "abliteration_complete").write_text(json.dumps(result) + "\n")
        return result

    modified = 0
    written: list[Path] = []
    weight_map: dict[str, str] = {}
    split = False
    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors shards in {model_dir}")

    def write_part(name: str, part: dict) -> None:
        out_shard = dst / name
        written.append(out_shard)
        mx.save_safetensors(str(out_shard), part)
        weight_map.update(dict.fromkeys(part, name))
        part.clear()
        mx.clear_cache()

    try:
        for shard in shards:
            tensors = mx.load(str(shard))  # lazy: nothing read until evaluated
            targets = {k: a for k in tensors
                       if (c := _component(k)) and (a := alpha_of(*c)) != 0}
            if not targets:
                out_shard = dst / shard.name
                written.append(out_shard)
                shutil.copy2(shard, out_shard)
                weight_map.update(dict.fromkeys(tensors, shard.name))
                del tensors
                continue
            split = True
            part: dict = {}
            part_bytes, n_part = 0, 0
            for key in sorted(tensors):
                if key in targets:
                    param = key[: -len(".weight")]
                    if param + ".scales" in tensors:
                        _modify_quantized(tensors, key, param, dirs, targets[key], quant)
                    else:
                        _modify_raw(tensors, key, dirs, targets[key])
                    modified += 1
            # pop as we go: an array still held by `tensors` stays resident
            # after its part is written
            for key in sorted(tensors):
                part[key] = tensors.pop(key)
                part_bytes += part[key].nbytes
                if part_bytes >= PART_BYTES:
                    n_part += 1
                    write_part(f"{shard.stem}-part{n_part}.safetensors", part)
                    part_bytes = 0
            if part:
                n_part += 1
                write_part(f"{shard.stem}-part{n_part}.safetensors", part)
            del tensors
            mx.clear_cache()

        for f in sorted(src.iterdir()):
            if f.is_file() and f.suffix != ".safetensors":
                if split and f.name == "model.safetensors.index.json":
                    continue
                out_f = dst / f.name
                written.append(out_f)
                shutil.copy2(f, out_f)
        if split:
            index = dst / "model.safetensors.index.json"
            written.append(index)
            index.write_text(json.dumps(
                {"metadata": {}, "weight_map": dict(sorted(weight_map.items()))},
                indent=2) + "\n")
    except BaseException:
        for p in written:
            p.unlink(missing_ok=True)
        if not dst_existed:
            shutil.rmtree(dst, ignore_errors=True)
        raise

    result = {
        "modified_tensors": modified,
        **meta,
        "out_dir": out_dir,
        "quantized": quant is not None,
        "shards": len(shards),
    }
    # only mark the directory complete once every shard and aux file landed
    (dst / "abliteration_complete").write_text(json.dumps(result) + "\n")
    return result
