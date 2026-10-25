"""Mixed-precision requantization of an already-quantized MLX model.

Drops MLP projections (gate/up/down — about two thirds of a dense layer) of
the middle layers to fewer bits, keeping attention, embeddings, head, and the
first/last `keep_edge` layers at the source precision, so a model can fit in
memory that otherwise only streams. Works tensor by tensor on lazily loaded
shards (dequantize -> quantize), written in ~1GB parts like abliteration.
Requantizing from 3-bit compounds error; a bf16 source would be better but is
5x the download.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import mlx.core as mx

from .abliteration.apply import PART_BYTES, _quant_params

MLP_RE = re.compile(r"^(?:language_model\.)?model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj$")


def requantize(src_dir: str, out_dir: str, mlp_bits: int = 2,
               group_size: int = 64, keep_edge: int = 4) -> dict:
    src, dst = Path(src_dir), Path(out_dir)
    if dst.exists() and dst.resolve() == src.resolve():
        raise ValueError("out_dir must differ from src_dir")
    cfg = json.loads((src / "config.json").read_text())
    quant = _quant_params(src_dir)
    if not quant:
        raise ValueError(f"{src_dir} is not quantized")
    text = cfg.get("text_config", cfg)
    n_layers = text["num_hidden_layers"]

    def target(param: str) -> bool:
        m = MLP_RE.match(param)
        return bool(m) and keep_edge <= int(m.group(1)) < n_layers - keep_edge

    dst.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}
    overrides: dict[str, dict] = {}
    changed, before, after = 0, 0, 0
    for shard in sorted(src.glob("model*.safetensors")):
        tensors = mx.load(str(shard))
        for key in [k for k in tensors if k.endswith(".weight")]:
            param = key[: -len(".weight")]
            if not target(param) or param + ".scales" not in tensors:
                continue
            spec = {**quant, **quant.get(param, {})}
            w = mx.dequantize(tensors[key], tensors[param + ".scales"],
                              tensors[param + ".biases"], group_size=int(spec["group_size"]),
                              bits=int(spec["bits"]), dtype=mx.float32)
            dtype = tensors[param + ".scales"].dtype
            before += sum(tensors[param + s].nbytes for s in (".weight", ".scales", ".biases"))
            wq, s, b = mx.quantize(w, group_size=group_size, bits=mlp_bits)
            tensors[key], tensors[param + ".scales"], tensors[param + ".biases"] = (
                wq, s.astype(dtype), b.astype(dtype))
            after += sum(tensors[param + s].nbytes for s in (".weight", ".scales", ".biases"))
            overrides[param] = {"group_size": group_size, "bits": mlp_bits}
            changed += 1
        part: dict = {}
        part_bytes, n_part = 0, 0
        for key in sorted(tensors):
            part[key] = tensors.pop(key)
            part_bytes += part[key].nbytes
            if part_bytes >= PART_BYTES or not tensors:
                n_part += 1
                name = f"{shard.stem}-part{n_part}.safetensors"
                mx.save_safetensors(str(dst / name), part)
                weight_map.update(dict.fromkeys(part, name))
                part.clear()
                part_bytes = 0
                mx.clear_cache()
        del tensors

    for f in sorted(src.iterdir()):
        if f.is_file() and f.suffix != ".safetensors" and f.name not in (
                "model.safetensors.index.json", "config.json"):
            shutil.copy2(f, dst / f.name)
    q = {**quant, **overrides}
    cfg["quantization"] = q
    if "quantization_config" in cfg:
        cfg["quantization_config"] = q
    (dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (dst / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {}, "weight_map": dict(sorted(weight_map.items()))}, indent=2) + "\n")
    size = sum(f.stat().st_size for f in dst.glob("*.safetensors")) / 2**30
    return {"requantized": changed, "saved_gb": (before - after) / 2**30,
            "size_gb": size, "out_dir": out_dir}
