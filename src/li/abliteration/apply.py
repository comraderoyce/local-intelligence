"""Apply abliteration by orthogonalizing decoder output weights in place.

For every captured layer >= `layer`, the residual-output projections
(self_attn.o_proj, mlp.down_proj, MoE expert/shared down_proj) are modified as
W' = (I - alpha * d d^T / (d.d)) W, removing the component of each output row
along the refusal direction d. Quantized tensors are dequantized, modified in
float32, and requantized with the model's own bits/group_size; shards are
streamed one at a time and rewritten, never holding more than one shard.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors.numpy import safe_open as np_safe_open

# Shard IO uses mx.load/mx.save_safetensors: mlx-community shards carry bf16
# scales/biases, which the safetensors numpy and mlx readers in this env
# cannot materialize (TypeError: data type 'bfloat16' not understood).

TARGET_RE = re.compile(
    r"^model\.layers\.(\d+)\."
    r"(self_attn\.o_proj|mlp\.down_proj|mlp\.experts\.\d+\.down_proj|"
    r"mlp\.shared_expert\.down_proj)\.weight$"
)


def _load_directions(stats_path: str) -> np.ndarray:
    with np_safe_open(stats_path, framework="numpy") as f:
        return f.get_tensor("directions").astype(np.float32)


def _quant_params(model_dir: str) -> dict | None:
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    return cfg.get("quantization") or cfg.get("quantization_config")


def _project_out(w: mx.array, d_proj: mx.array, alpha: float) -> mx.array:
    """W' = W - alpha * (d d^T / (d.d)) W for W [out, in]; d_proj = d / d.d."""
    assert w.shape[0] == d_proj.shape[-1], (
        f"direction dim {d_proj.shape[-1]} != weight rows {w.shape[0]}")
    orig_dtype = w.dtype
    assert str(orig_dtype) not in ("uint32", "uint16", "uint8", "int32"), (
        f"{orig_dtype} looks like packed quantized codes, not raw weights")
    w32 = w.astype(mx.float32)
    proj = d_proj @ w32                        # [in]
    out = w32 - alpha * d_proj[:, None] * proj[None, :]
    return out.astype(orig_dtype)


def _modify_quantized(tensors: dict, key: str, param: str,
                      d_proj: mx.array, alpha: float, quant: dict) -> None:
    spec = {**quant, **quant.get(param, {})}
    bits, group_size = int(spec["bits"]), int(spec["group_size"])
    # shard layout: <param>.weight is packed, alongside <param>.scales/.biases
    scales, biases = tensors[param + ".scales"], tensors[param + ".biases"]
    w = mx.dequantize(tensors[key], scales, biases,
                      group_size=group_size, bits=bits, dtype=mx.float32)
    w_new = _project_out(w, d_proj, alpha)
    wq, s, b = mx.quantize(w_new, group_size=group_size, bits=bits)
    tensors[key] = wq
    tensors[param + ".scales"] = s.astype(scales.dtype)
    tensors[param + ".biases"] = b.astype(biases.dtype)


def _modify_raw(tensors: dict, key: str, d_proj: mx.array, alpha: float) -> None:
    tensors[key] = _project_out(tensors[key], d_proj, alpha)


def apply_abliteration(model_dir: str, stats_path: str, layer: int,
                       alpha: float, out_dir: str) -> dict:
    src = Path(model_dir)
    dst = Path(out_dir)
    if dst.exists() and dst.resolve() == src.resolve():
        raise ValueError("out_dir must differ from model_dir")
    dst_existed = dst.exists()
    dst.mkdir(parents=True, exist_ok=True)

    directions_np = _load_directions(stats_path)
    quant = _quant_params(model_dir)
    # canonical abliteration: one direction, chosen at `layer`, orthogonalized
    # out of every residual-output projection from that layer upward
    d = mx.array(directions_np[layer])
    d_proj = d / (d @ d)  # constant factor of (d d^T / d.d) W

    modified = 0
    written: list[Path] = []
    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors shards in {model_dir}")
    try:
        for shard in shards:
            out_shard = dst / shard.name
            written.append(out_shard)
            tensors = dict(mx.load(str(shard)))
            targets = [(k, m) for k in tensors if (m := TARGET_RE.match(k))]
            if not targets:
                del tensors
                mx.clear_cache()
                shutil.copy2(shard, out_shard)
                continue
            for key, m in targets:
                if int(m.group(1)) < layer:
                    continue
                param = key[: -len(".weight")]
                if param + ".scales" in tensors:
                    _modify_quantized(tensors, key, param, d_proj, alpha, quant)
                else:
                    _modify_raw(tensors, key, d_proj, alpha)
                modified += 1
            mx.save_safetensors(str(out_shard), tensors)
            del tensors
            mx.clear_cache()

        for f in sorted(src.iterdir()):
            if f.is_file() and f.suffix != ".safetensors":
                out_f = dst / f.name
                written.append(out_f)
                shutil.copy2(f, out_f)
    except BaseException:
        for p in written:
            p.unlink(missing_ok=True)
        if not dst_existed:
            shutil.rmtree(dst, ignore_errors=True)
        raise

    result = {
        "modified_tensors": modified,
        "layer": layer,
        "alpha": alpha,
        "out_dir": out_dir,
        "quantized": quant is not None,
        "shards": len(shards),
    }
    # only mark the directory complete once every shard and aux file landed
    (dst / "abliteration_complete").write_text(json.dumps(result) + "\n")
    return result
