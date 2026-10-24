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
    # prompt / generation budgets
    n_harmful: int = 128
    n_harmless: int = 128
    n_advbench_eval: int = 100
    n_xstest_eval: int = 100
    max_seq: int = 384
    max_new_tokens: int = 256
    seed: int = 0
    # training
    train_data: str = ""
    lora_iters: int = 1000
    lora_lr: float = 1e-4
    lora_batch_size: int = 1
    lora_layers: int = 16
    fused_dir: str = ""

    @property
    def run_dir(self) -> str:
        return f"runs/{self.name}-{self.quant_bits}bit"

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
    return Config(**{k: v for k, v in data.items() if k in known})
