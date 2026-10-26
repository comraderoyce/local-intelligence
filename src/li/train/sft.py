"""SFT data normalization and LoRA training/fusion via mlx_lm subprocesses.

prepare_sft_data writes a --data directory of {train, valid}.jsonl for
mlx_lm lora (mlx_lm/tuner/datasets.py::load_local_dataset) in one of two
formats, since mlx_lm picks a single dataset type from the first row:

- chat: question/answer rows from .jsonl/.json/.csv, normalized to
  {"messages": [...]} and trained on the answers only (--mask-prompt);
- text: unstructured documents (.txt/.md files, a directory of them, or
  {"text": ...} rows), packed into chunks of at most max_seq tokens, since
  mlx_lm truncates longer sequences rather than splitting them.

Note: the installed mlx_lm (0.32.2) has no `--fuse` flag on `mlx_lm lora`;
adapter fusion is the separate `mlx_lm fuse` subcommand.
"""
from __future__ import annotations

import csv
import json
import random
import re
import subprocess
import sys
from collections import deque
from pathlib import Path

from ..config import Config

_VALID_ROLES = {"system", "user", "assistant"}
_COMPLETION_KEYS = ("output", "response", "answer")
_TEXT_SUFFIXES = (".txt", ".md")


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

    if isinstance(row.get("text"), str):
        return {"text": row["text"]}

    raise SystemExit(
        f"{where}: row matches no supported schema.\n"
        "Accepted schemas:\n"
        '  {"instruction": str, "output"|"response"|"answer": str}\n'
        '  {"prompt": str, "completion": str}\n'
        '  {"messages": [{"role": str, "content": str}, ...]}\n'
        '  {"text": str}   (unstructured text)'
    )


def _read_rows(src: str) -> list[dict]:
    path = Path(src)
    if not path.exists():
        raise SystemExit(f"no such file: {src}")
    if path.is_dir():
        docs = sorted(f for f in path.rglob("*")
                      if f.is_file() and f.suffix.lower() in _TEXT_SUFFIXES)
        if not docs:
            raise SystemExit(f"{src}: no {'/'.join(_TEXT_SUFFIXES)} files")
        return [{"text": f.read_text()} for f in docs]
    suffix = path.suffix.lower()
    if suffix in _TEXT_SUFFIXES:
        return [{"text": path.read_text()}]
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
    raise SystemExit(f"{src}: unsupported type {suffix!r} — use .jsonl, .json, .csv, "
                     ".txt, .md or a directory of text files")


def chunk_text(text: str, tokenizer, max_tokens: int) -> list[str]:
    """Pack paragraphs into chunks of at most ~max_tokens tokens; a paragraph
    longer than that is split on token boundaries."""
    limit = max(16, int(max_tokens * 0.95) - 1)   # margin for joins and EOS
    chunks, cur, cur_len = [], [], 0
    for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not para:
            continue
        ids = tokenizer.encode(para)
        pieces = ([para] if len(ids) <= limit else
                  [tokenizer.decode(ids[i:i + limit]) for i in range(0, len(ids), limit)])
        for piece in pieces:
            n = len(ids) if len(pieces) == 1 else limit
            if cur and cur_len + n > limit:
                chunks.append("\n\n".join(cur))
                cur, cur_len = [], 0
            cur.append(piece)
            cur_len += n
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def prepare_sft_data(src: str, out_dir: str, tokenizer=None,
                     max_seq: int = 512) -> dict:
    """Normalize src to chat or text jsonl, shuffle (seed 0), split 95/5.
    Text needs the model's `tokenizer` to chunk documents to max_seq tokens."""
    rows = _read_rows(src)
    kinds = {"text" if "text" in r else "chat" for r in rows}
    if len(kinds) > 1:
        raise SystemExit(f"{src} mixes question/answer rows and text rows; mlx_lm trains "
                         "one format per run. Train on the text first, then on the "
                         "question/answer data (see README: Fine-tuning)")
    kind = kinds.pop()
    if kind == "text":
        if tokenizer is None:
            raise ValueError("text data needs a tokenizer to chunk")
        rows = [{"text": c} for r in rows for c in chunk_text(r["text"], tokenizer, max_seq)]
    if len(rows) < 2:
        raise SystemExit(f"{src}: need at least 2 examples (got {len(rows)})")
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
    return {"train": len(rows) - n_valid, "valid": n_valid, "kind": kind}


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
            batch_size: int, num_layers: int, adapter_dir: str,
            max_seq: int = 512, mask_prompt: bool = False) -> None:
    """LoRA fine-tune model_dir on data_dir via `mlx_lm lora`. mask_prompt
    trains on assistant turns only (chat data; mlx_lm rejects it for text)."""
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
            "--max-seq-length", str(max_seq),
            "--adapter-path", adapter_dir,
            "--seed", str(cfg.seed),
            *(["--mask-prompt"] if mask_prompt else []),
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
