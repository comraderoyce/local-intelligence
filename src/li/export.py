"""Export an abliteration adapter as a standalone model directory.

On quantized bases abliteration is a rank-k adapter, not edited weights:
requantizing a small edit at the base's own bits snaps most of it back onto
the original grid (see abliteration/apply.py). Export instead applies the
edit to the original bf16 weights and quantizes once afterwards, where the
rounding error is independent of the edit rather than biased back to the
unedited weights.

The edit is recovered from the adapter itself, so any adapter `abliterate` or
`optimize` wrote works: each module stores lora_b = -a D (orthonormal rows D
scaled by strength a), and the same correction on a bf16 weight is

    W' = W + lora_b^T (D W)          (residual writers, W [out, in])
    E' = E + (E D^T) lora_b          (token embedding, E [vocab, D])

Shards are read lazily and rewritten in ~1GB parts, then `mlx_lm.convert`
quantizes the edited copy, so peak memory stays at a few GB on any size model.
"""
from __future__ import annotations

import gc
import json
import shutil
from pathlib import Path

import mlx.core as mx

from .abliteration.apply import EMBED_CHUNK, PART_BYTES

SOURCE_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.jinja", "*.model",
                   "tokenizer*", "merges.txt", "vocab.json"]


def download_source(repo: str, local_dir: str) -> Path:
    """Original (bf16) HF weights, downloaded once and reused by later exports."""
    target = Path(local_dir)
    # Always ask the hub: it resumes an interrupted download and returns at once
    # when complete. Checking for config.json alone treated a partial download
    # (config arrives first) as done.
    from huggingface_hub import snapshot_download
    print(f"pulling source weights {repo} -> {target} (bf16; large, one-time)")
    snapshot_download(repo_id=repo, local_dir=target, allow_patterns=SOURCE_PATTERNS)
    index = target / "model.safetensors.index.json"
    if index.exists():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        missing = sorted(s for s in shards if not (target / s).exists())
        if missing:
            raise SystemExit(f"source download incomplete: missing {', '.join(missing)}")
    return target


def _edits(adapter_dir: str) -> tuple[dict[str, tuple[mx.array, mx.array]], dict]:
    """param -> (D [k, out], lora_b [k, out]) from an abliteration adapter."""
    cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    meta = cfg.get("abliteration")
    if meta is None:
        raise SystemExit(f"{adapter_dir} is not an abliteration adapter "
                         "(use `li fuse` for trained LoRA adapters)")
    if meta.get("row_normalization", "none") != "none":
        raise SystemExit("export of row-normalized abliteration is not supported "
                         "(its lora_b rows do not encode a unit direction)")
    tensors = mx.load(str(Path(adapter_dir) / "adapters.safetensors"))
    edits = {}
    for key, b in tensors.items():
        if not key.endswith(".lora_b"):
            continue
        b = b.astype(mx.float32)
        a = mx.sqrt(mx.sum(b * b, axis=-1, keepdims=True))   # strength per row
        if float(a.max().item()) == 0.0:
            continue
        edits[key[: -len(".lora_b")]] = (-b / mx.maximum(a, 1e-12), b)
    return edits, cfg


def _edit_linear(w: mx.array, d: mx.array, b: mx.array) -> mx.array:
    w32 = w.astype(mx.float32)
    return (w32 + b.T @ (d @ w32)).astype(w.dtype)


def _edit_embedding(e: mx.array, d: mx.array, b: mx.array) -> mx.array:
    rows = []
    for r in range(0, e.shape[0], EMBED_CHUNK):
        chunk = e[r : r + EMBED_CHUNK].astype(mx.float32)
        rows.append((chunk + (chunk @ d.T) @ b).astype(e.dtype))
        mx.eval(rows[-1])
    return mx.concatenate(rows)


def _param_key(key: str, edits: dict) -> str | None:
    """Adapter param for a source tensor key. MLX checkpoints of multimodal
    models drop HF's `model.language_model.` wrapper, so try both spellings."""
    if not key.endswith(".weight"):
        return None
    param = key[: -len(".weight")]
    for cand in (param, param.replace("model.language_model.", "language_model.model."),
                 param.replace("model.language_model.", "model.")):
        if cand in edits:
            return cand
    return None


def write_edited(source_dir: str, edits: dict, out_dir: str) -> int:
    """Copy of `source_dir` with every adapter edit applied to its weights."""
    src, dst = Path(source_dir), Path(out_dir)
    dst.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    text = cfg.get("text_config", cfg)
    tied = bool(cfg.get("tie_word_embeddings", text.get("tie_word_embeddings", False)))
    weight_map: dict[str, str] = {}
    done: set[str] = set()
    for shard in sorted(src.glob("*.safetensors")):
        tensors = mx.load(str(shard))
        for key in list(tensors):
            param = _param_key(key, edits)
            if param is None:
                continue
            d, b = edits[param]
            if param.endswith("embed_tokens"):
                if tied:
                    # the output head shares these weights; keep it unedited, as
                    # the adapter does, by untying it into its own tensor
                    tensors["lm_head.weight"] = tensors[key]
                tensors[key] = _edit_embedding(tensors[key], d, b)
            else:
                tensors[key] = _edit_linear(tensors[key], d, b)
            done.add(param)
        part: dict = {}
        part_bytes, n_part = 0, 0
        for key in sorted(tensors):
            part[key] = tensors.pop(key)
            part_bytes += part[key].nbytes
            if part_bytes >= PART_BYTES or not tensors:
                n_part += 1
                name = f"{shard.stem}-part{n_part}.safetensors"
                mx.save_safetensors(str(dst / name), part, metadata={"format": "pt"})
                weight_map.update(dict.fromkeys(part, name))
                part.clear()
                part_bytes = 0
                mx.clear_cache()
        del tensors
    missing = sorted(set(edits) - done)
    if missing:
        raise SystemExit(f"{len(missing)} adapter modules have no matching source weight "
                         f"(e.g. {missing[0]}); is the source the same model?")
    for f in sorted(src.iterdir()):
        if f.is_file() and f.suffix != ".safetensors" and f.name not in (
                "model.safetensors.index.json", "config.json"):
            shutil.copy2(f, dst / f.name)
    if tied and any(p.endswith("embed_tokens") for p in done):
        cfg["tie_word_embeddings"] = False
        if "text_config" in cfg:
            cfg["text_config"]["tie_word_embeddings"] = False
    (dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (dst / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {}, "weight_map": dict(sorted(weight_map.items()))}, indent=2) + "\n")
    return len(done)


def _convert(src: str, out: str, quantize: bool, bits: int, group_size: int) -> None:
    """mlx_lm.convert, saving in ~1GB shards. Its default 5GB shard holds a
    whole quantized 8B, and saving a shard evaluates every source tensor
    feeding it at once: all 16GB of bf16 on a 16GB Mac. Small shards bound
    the peak to the quantized weights saved so far plus about one shard."""
    from mlx_lm import convert, utils
    make_shards = utils.make_shards
    utils.make_shards = lambda weights, max_file_size_gb=1: make_shards(weights, 1)
    try:
        convert(src, mlx_path=out, quantize=quantize,
                q_bits=bits if quantize else None,
                q_group_size=group_size if quantize else None, dtype="bfloat16")
    finally:
        utils.make_shards = make_shards
        # convert's model is gone, but MLX keeps its buffers cached; a model
        # loaded next (export verification) would otherwise sit on top of them
        gc.collect()
        mx.clear_cache()


def export_model(adapter_dir: str, repo: str, source_dir: str, out_dir: str,
                 bits: int, group_size: int, keep_edited: bool = False) -> dict:
    out = Path(out_dir)
    if out.exists():
        raise SystemExit(f"{out_dir} exists — remove it or pass --out")
    edits, adapter_cfg = _edits(adapter_dir)
    source = download_source(repo, source_dir)
    edited = out.with_name(out.name + ".bf16-edited")
    if edited.exists():
        shutil.rmtree(edited)
    print(f"applying {len(edits)} module edits to {source} -> {edited}")
    n = write_edited(str(source), edits, str(edited))

    quantize = bits < 16
    print(f"converting -> {out} ({f'{bits}-bit, group {group_size}' if quantize else 'bf16'})")
    try:
        _convert(str(edited), str(out), quantize, bits, group_size)
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)   # an empty out dir blocks a rerun
        raise
    if not keep_edited:
        shutil.rmtree(edited)
    info = {"source": repo, "adapter": adapter_dir, "edited_modules": n,
            "bits": bits if quantize else 16, "group_size": group_size if quantize else None,
            "abliteration": adapter_cfg.get("abliteration")}
    (out / "li_export.json").write_text(json.dumps(info, indent=2) + "\n")
    size = sum(f.stat().st_size for f in out.glob("*.safetensors")) / 2**30
    return {**info, "out_dir": out_dir, "size_gb": size}


def refusal_opener_prob(model_dir: str, prompts: list[str]) -> float:
    """Mean probability that the first response token opens a refusal."""
    from .abliteration.select import _first_token_logprobs, _refusal_token_ids
    from .stream import load_for_inference
    model, tokenizer, _ = load_for_inference(model_dir, stream=False)
    lp = _first_token_logprobs(model, tokenizer, prompts, None)
    ids = mx.array(_refusal_token_ids(tokenizer))
    p = mx.exp(lp[:, ids]).sum(axis=-1).mean().item()
    del model, tokenizer
    mx.clear_cache()
    return p
