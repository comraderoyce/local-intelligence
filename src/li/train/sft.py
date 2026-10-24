"""SFT data normalization and LoRA training/fusion via mlx_lm subprocesses.

prepare_sft_data normalizes .jsonl/.json/.csv into OpenAI chat-format jsonl
({"messages": [{"role": ..., "content": ...}, ...]}), which is exactly the
ChatDataset format mlx_lm lora expects in a --data directory of
{train, valid}.jsonl (mlx_lm/tuner/datasets.py::load_local_dataset).

Note: the installed mlx_lm (0.32.2) has no `--fuse` flag on `mlx_lm lora`;
adapter fusion is the separate `mlx_lm fuse` subcommand.
"""
from __future__ import annotations

import csv
import json
import random
import subprocess
import sys
from collections import deque
from pathlib import Path

from ..config import Config

_VALID_ROLES = {"system", "user", "assistant"}
_COMPLETION_KEYS = ("output", "response", "answer")


def _chat_row(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def _normalize_row(row: dict, where: str) -> dict:
    """Map one source row to chat format; SystemExit on unknown schema."""
    msgs = row.get("messages")
    if isinstance(msgs, list) and msgs:
        if all(
            isinstance(m, dict)
            and m.get("role") in _VALID_ROLES
            and isinstance(m.get("content"), str)
            for m in msgs
        ):
            return {"messages": [{"role": m["role"], "content": m["content"]} for m in msgs]}

    if isinstance(row.get("instruction"), str) and not isinstance(row.get("prompt"), str):
        assistant = next((row[k] for k in _COMPLETION_KEYS if isinstance(row.get(k), str)), None)
        if assistant is not None:
            return _chat_row(row["instruction"], assistant)

    if isinstance(row.get("prompt"), str) and isinstance(row.get("completion"), str):
        return _chat_row(row["prompt"], row["completion"])

    raise SystemExit(
        f"{where}: row matches no supported schema.\n"
        "Accepted schemas:\n"
        '  {"instruction": str, "output"|"response"|"answer": str}\n'
        '  {"prompt": str, "completion": str}\n'
        '  {"messages": [{"role": str, "content": str}, ...]}'
    )


def _read_rows(src: str) -> list[dict]:
    path = Path(src)
    if not path.exists():
        raise SystemExit(f"no such file: {src}")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip():
                rows.append(_normalize_row(json.loads(line), f"{src}:{i}"))
        return rows
    if suffix == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            raise SystemExit(f"{src}: .json must contain a list of row objects")
        return [_normalize_row(r, f"{src}:{i}") for i, r in enumerate(data, 1)]
    if suffix == ".csv":
        with open(path, newline="") as fid:
            return [
                _normalize_row(dict(r), f"{src}:row{i}")
                for i, r in enumerate(csv.DictReader(fid), 1)
            ]
    raise SystemExit(f"{src}: unsupported type {suffix!r} — use .jsonl, .json or .csv")


def prepare_sft_data(src: str, out_dir: str) -> dict:
    """Normalize src to chat-format jsonl, shuffle (seed 0), split 95/5."""
    rows = _read_rows(src)
    random.Random(0).shuffle(rows)
    n_valid = max(1, int(len(rows) * 0.05))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows[n_valid:])
    )
    (out / "valid.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows[:n_valid])
    )
    return {"train": len(rows) - n_valid, "valid": n_valid}


def _run_streamed(cmd: list[str], what: str) -> None:
    """Run cmd, printing output live; raise with the tail on failure."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    assert proc.stdout is not None
    tail: deque[str] = deque(maxlen=30)
    for line in proc.stdout:
        print(line, end="")
        tail.append(line)
    if proc.wait() != 0:
        raise RuntimeError(
            f"{what} failed (exit {proc.returncode}); last output:\n{''.join(tail)}"
        )


def run_sft(model_dir: str, data_dir: str, cfg: Config, iters: int, lr: float,
            batch_size: int, num_layers: int, adapter_dir: str) -> None:
    """LoRA fine-tune model_dir on data_dir via `mlx_lm lora`."""
    Path(adapter_dir).mkdir(parents=True, exist_ok=True)
    _run_streamed(
        [
            sys.executable, "-m", "mlx_lm", "lora",
            "--train",
            "--model", model_dir,
            "--data", data_dir,
            "--fine-tune-type", "lora",
            "--num-layers", str(num_layers),
            "--batch-size", str(batch_size),
            "--iters", str(iters),
            "--learning-rate", str(lr),
            "--max-seq-length", str(cfg.max_seq),
            "--adapter-path", adapter_dir,
            "--seed", str(cfg.seed),
        ],
        "lora training",
    )


def fuse_adapter(model_dir: str, adapter_dir: str, out_dir: str) -> None:
    """Fuse a trained LoRA adapter into a standalone model dir."""
    _run_streamed(
        [
            sys.executable, "-m", "mlx_lm", "fuse",
            "--model", model_dir,
            "--adapter-path", adapter_dir,
            "--save-path", out_dir,
        ],
        "adapter fusion",
    )
