"""Model acquisition: pull pre-quantized MLX repos or convert fp16 locally."""
from __future__ import annotations

import json
from pathlib import Path

from .config import Config
from .registry import spec_for


def download_draft(cfg: Config) -> None:
    """Pull the registry's speculative-decoding draft model, if it has one."""
    spec = spec_for(cfg.name, cfg.repo)
    if not spec.draft_repo or cfg.draft_dir:
        return
    from huggingface_hub import snapshot_download
    cfg.draft_dir = f"models/{spec.draft_repo.split('/')[-1].lower()}"
    if not (Path(cfg.draft_dir) / "config.json").exists():
        print(f"pulling draft model {spec.draft_repo} -> {cfg.draft_dir}")
        snapshot_download(repo_id=spec.draft_repo, local_dir=cfg.draft_dir)


def download_model(cfg: Config) -> Path:
    target = Path(cfg.model_dir)
    if (target / "config.json").exists():
        print(f"model already present: {target}")
        return target

    from huggingface_hub import snapshot_download
    spec = spec_for(cfg.name, cfg.repo)
    mlx_repo = spec.mlx_repo
    if mlx_repo:
        print(f"pulling pre-quantized {mlx_repo} -> {target}")
        snapshot_download(repo_id=mlx_repo, local_dir=target)
    else:
        from mlx_lm import convert
        print(f"converting {cfg.repo} -> {target} ({cfg.quant_bits}-bit, group {cfg.quant_group})")
        if spec.params_b >= 13:
            print("WARNING: fp16 source download is large (13B+); consider a "
                  "mlx-community pre-quant repo instead")
        convert(cfg.repo, mlx_path=str(target), quantize=True,
                q_bits=cfg.quant_bits, q_group_size=cfg.quant_group)

    info = json.loads((target / "config.json").read_text())
    q = info.get("quantization")
    size_gb = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 2**30
    print(f"done: {size_gb:.1f}GB, quantization={'/'.join(str(q[k]) for k in ('bits', 'group_size')) if q else 'none'}")
    return target
