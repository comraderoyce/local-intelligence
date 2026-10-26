"""Model acquisition: pull pre-quantized MLX repos or convert fp16 locally."""
from __future__ import annotations

import json
from pathlib import Path

from .config import Config
from .registry import JUDGE_DIR, JUDGE_REPO, spec_for


def _pull(repo: str, target: Path, what: str) -> None:
    """Always ask the hub: it resumes an interrupted download and returns at
    once when complete. Checking for config.json alone treated a partial
    download (config arrives first) as done. Offline, a local copy is used
    as-is with a warning."""
    from huggingface_hub import snapshot_download
    print(f"pulling {what} {repo} -> {target}")
    try:
        snapshot_download(repo_id=repo, local_dir=target)
    except Exception as e:
        missing = missing_weights(target)
        if missing:
            raise SystemExit(f"couldn't reach the hub for {repo} ({type(e).__name__}) "
                             f"and {target} is incomplete: missing {', '.join(missing)}")
        print(f"warning: couldn't check {repo} against the hub ({type(e).__name__}); "
              f"using {target} as-is")


def missing_weights(target: Path) -> list[str]:
    """Files a model dir still needs: config, and every shard its index names
    (or at least one shard when unsharded)."""
    if not (target / "config.json").exists():
        return ["config.json"]
    index = target / "model.safetensors.index.json"
    if index.exists():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        return sorted(s for s in shards if not (target / s).exists())
    return [] if any(target.glob("*.safetensors")) else ["*.safetensors"]


def download_draft(cfg: Config) -> None:
    """Pull the registry's speculative-decoding draft model, if it has one."""
    spec = spec_for(cfg.name, cfg.repo)
    default = f"models/{spec.draft_repo.split('/')[-1].lower()}"
    if not spec.draft_repo or cfg.draft_dir not in ("", default):
        return  # none, or the user pointed draft_dir elsewhere
    cfg.draft_dir = default
    _pull(spec.draft_repo, Path(default), "draft model")


def download_judge(cfg: Config) -> None:
    """Pull the default refusal judge when the config points at it."""
    if cfg.judge_dir != JUDGE_DIR:
        return
    _pull(JUDGE_REPO, Path(cfg.judge_dir), "refusal judge")


def download_model(cfg: Config) -> Path:
    target = Path(cfg.model_dir)
    spec = spec_for(cfg.name, cfg.repo)
    mlx_repo = spec.mlx_repo
    if mlx_repo:
        _pull(mlx_repo, target, "pre-quantized model")
    elif (target / "tokenizer_config.json").exists():
        # mlx_lm.convert writes weights, then config, then tokenizer files
        print(f"model already present: {target}")
        return target
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
