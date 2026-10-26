"""Prompt corpora for direction extraction and refusal evals.

Sources, all public:
- harmful:  AdvBench (520 harmful behaviors)
- harmless: Alpaca instruction subset (benign tasks)
- over-refusal probe: XSTest (exaggerated-safety prompts)

Cached under datasets/ by the HF datasets library.
"""
from __future__ import annotations

import random

ADV_BENCH_HF = "walledai/AdvBench"
ADV_BENCH_CSV = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_behaviors.csv"
)
HARMLESS_HF = "tatsu-lab/alpaca"
XSTEST_HF = "walledai/XSTest"

# Fixed, non-overlapping slices of the once-shuffled corpora.  Changing the
# requested sample count must not silently move examples between development
# and evaluation (the old seed, seed+1 scheme did exactly that: shuffled
# prefixes from different seeds still overlap heavily).
SPLITS: dict[str, tuple[int, int | None]] = {
    "measure": (0, 128),
    "select": (128, 160),
    "optimize": (160, 224),
    "eval": (224, None),
}


def _sample(prompts: list[str], n: int, seed: int, split: str | None = None) -> list[str]:
    seen, uniq = set(), []
    for p in prompts:
        p = " ".join(str(p).split())
        if p and p not in seen:
            seen.add(p)
            uniq.append(p)
    rng = random.Random(seed)
    rng.shuffle(uniq)
    start, stop = SPLITS.get(split, (0, None))
    available = len(uniq) if stop is None else min(stop, len(uniq))
    if start + n > available:
        name = split or "default"
        raise ValueError(
            f"requested {n} prompts from {name!r} split, but only "
            f"{max(0, available - start)} are reserved; reduce n or revise SPLITS"
        )
    return uniq[start:start + n]


def load_advbench(n: int = 128, seed: int = 0,
                  split: str | None = None) -> list[str]:
    from datasets import load_dataset
    try:
        rows = load_dataset(ADV_BENCH_HF, split="train")
        prompts = rows["prompt"]
    except Exception:
        rows = load_dataset("csv", data_files=ADV_BENCH_CSV, split="train")
        prompts = rows["goal"]
    return _sample(list(prompts), n, seed, split)


def load_harmless(n: int = 128, seed: int = 0,
                  split: str | None = None) -> list[str]:
    from datasets import load_dataset
    rows = load_dataset(HARMLESS_HF, split="train")
    return _sample([r["instruction"] for r in rows], n, seed, split)


def load_xstest(n: int = 100, seed: int = 0) -> list[str]:
    from datasets import load_dataset
    try:
        rows = load_dataset(XSTEST_HF, split="train")
        prompts = rows["prompt"]
    except Exception:  # walledai/XSTest is gated; official safe prompts from the authors
        rows = load_dataset(
            "csv",
            data_files=(
                "https://raw.githubusercontent.com/paul-rottger/xstest/main/"
                "xstest_prompts.csv"
            ),
            split="train",
        )
        prompts = [
            r["prompt"] for r in rows if str(r.get("label", "safe")) == "safe"
        ]
    return _sample(list(prompts), n, seed)
