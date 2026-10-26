"""li.toml configuration: single source of truth for one model on one machine.

Flat key/value TOML. All paths are workspace-relative (resolved from cwd).
"""
from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import tomli_w

CONFIG_PATH = Path("li.toml")


@dataclass
class Config:
    # model identity
    name: str
    repo: str                    # HF source repo (provenance)
    model_dir: str               # local MLX (quantized) model dir
    quant_bits: int = 4
    quant_group: int = 64
    # abliteration artifacts
    stats_path: str = ""
    report_path: str = ""
    abliterated_dir: str = ""
    layer: int | None = None     # chosen refusal-direction layer
    layer_score: float | None = None
    alpha: float = 1.0           # direction removal strength, 1.0 = full
    pos: int | None = None       # template-suffix position (negative offset)
    from_layer: int = 0          # edit layers >= this; 0 = all (canonical)
    n_dirs: int = 1              # refusal directions removed (rank of the edit)
    embed_ablate: bool = True    # also orthogonalize the token embedding
    project_harmless: bool = False
    # prompt / generation budgets
    n_harmful: int = 128
    n_harmless: int = 128
    n_advbench_eval: int = 100
    n_xstest_eval: int = 100
    max_seq: int = 384
    max_new_tokens: int = 256
    # streamed inference (models larger than safe resident memory)
    stream_budget_gb: float = 0.0   # 0 = auto from available memory
    eval_batch: int = 8             # sequences per streamed forward pass
    judge_dir: str = ""              # dedicated local refusal classifier
    judge_kind: str = "qwen3guard"   # dedicated classifier prompt/parser
    draft_dir: str = ""             # speculative-decoding draft model
    num_draft: int = 3
    seed: int = 0
    # training
    train_data: str = ""
    lora_iters: int = 1000
    lora_lr: float = 1e-4
    lora_batch_size: int = 1
    lora_layers: int = 16
    fused_dir: str = ""
    export_dir: str = ""

    @property
    def run_dir(self) -> str:
        return f"runs/{self.name}-{self.quant_bits}bit"

    @property
    def export_path(self) -> str:
        """Standalone abliterated model (`li export`)."""
        return self.export_dir or f"{self.run_dir}/export"

    @property
    def source_dir(self) -> str:
        """Original unquantized weights, downloaded once for export."""
        return f"models/{self.name}-bf16"

    @property
    def eval_dir(self) -> str:
        return f"evals/{self.name}-{self.quant_bits}bit"

    @property
    def adapter_dir(self) -> str:
        return f"{self.run_dir}/adapters"


def save_config(cfg: Config, path: Path = CONFIG_PATH) -> None:
    data = {k: v for k, v in asdict(cfg).items() if v is not None}
    path.write_bytes(tomli_w.dumps(data).encode())


def load_config(path: Path = CONFIG_PATH) -> Config:
    if not path.exists():
        raise SystemExit(f"no {CONFIG_PATH} here — run `li init` first")
    data = tomllib.loads(path.read_text())
    known = {f.name for f in fields(Config)}
    unknown = sorted(set(data) - known)
    if unknown:
        # a warning, not an error: keys renamed or dropped between versions
        # shouldn't strand an existing workspace
        print(f"warning: ignoring unknown {path} keys: {', '.join(unknown)}")
    return Config(**{k: v for k, v in data.items() if k in known})
