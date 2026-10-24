"""Model registry with RAM-tier capability flags.

Every tier is honest about what fits where: abliteration needs only
weights + activations (forward passes + weight-space edits), training
needs headroom for adapters and optimizer state on top.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo: str            # HF source (fp16) — provenance / conversion source
    mlx_repo: str        # pre-quantized MLX repo to pull directly ("" = convert locally)
    params_b: float      # billions of params
    bits: int            # default quantization for this tier
    group: int           # quantization group size
    min_ram_gb: int      # floor for download + measure + abliterate
    train_ram_gb: int    # floor for local LoRA fine-tuning
    moe: bool = False
    note: str = ""


MODELS: dict[str, ModelSpec] = {
    "qwen3-0.6b": ModelSpec(
        "qwen3-0.6b", "Qwen/Qwen3-0.6B", "mlx-community/Qwen3-0.6B-4bit",
        0.6, 4, 64, 8, 8, note="pipeline smoke test tier",
    ),
    "qwen3-4b": ModelSpec(
        "qwen3-4b", "Qwen/Qwen3-4B", "mlx-community/Qwen3-4B-4bit",
        4.0, 4, 64, 8, 8,
    ),
    "qwen3-8b": ModelSpec(
        "qwen3-8b", "Qwen/Qwen3-8B", "mlx-community/Qwen3-8B-4bit",
        8.2, 4, 64, 8, 16, note="QLoRA fits 8GB tightly; comfortable at 16GB",
    ),
    "qwen3-14b": ModelSpec(
        "qwen3-14b", "Qwen/Qwen3-14B", "mlx-community/Qwen3-14B-4bit",
        14.8, 4, 64, 16, 32,
    ),
    "qwen3-30b-a3b": ModelSpec(
        "qwen3-30b-a3b", "Qwen/Qwen3-30B-A3B", "mlx-community/Qwen3-30B-A3B-3bit",
        30.5, 3, 64, 16, 64, moe=True,
        note="MoE; abliterate+inference on 16GB, training needs >=64GB",
    ),
    "qwen3-32b": ModelSpec(
        "qwen3-32b", "Qwen/Qwen3-32B", "mlx-community/Qwen3-32B-3bit",
        32.8, 3, 64, 16, 64,
        note="abliterate+inference on 16GB, training needs >=64GB",
    ),
}


def machine_ram_gb() -> int:
    page = os.sysconf("SC_PAGE_SIZE")
    nphys = os.sysconf("SC_PHYS_PAGES")
    return round(page * nphys / 2**30)


def resolve_spec(name_or_hf_id: str) -> ModelSpec:
    if name_or_hf_id in MODELS:
        return MODELS[name_or_hf_id]
    if "/" in name_or_hf_id:  # arbitrary HF id — fp16 convert path
        short = name_or_hf_id.split("/")[-1].lower()
        return ModelSpec(
            short, name_or_hf_id, "", 0.0, 4, 64, 8, 999,
            note="custom; training RAM unknown, convert pulls fp16 weights",
        )
    raise SystemExit(
        f"unknown model '{name_or_hf_id}'. registry: {', '.join(MODELS)} — "
        "or pass a HF id like 'org/model'"
    )


def fit_report(spec: ModelSpec, ram_gb: int) -> str:
    abl = "OK" if ram_gb >= spec.min_ram_gb else f"NO (needs >= {spec.min_ram_gb}GB)"
    train = "OK" if ram_gb >= spec.train_ram_gb else f"NO (needs >= {spec.train_ram_gb}GB)"
    moe = " [MoE]" if spec.moe else ""
    return (
        f"{spec.name:16s} {spec.params_b:>4.1f}B{moe:5s} {spec.bits}-bit  "
        f"abliterate: {abl:22s} lora-train: {train}  {spec.note}"
    )
