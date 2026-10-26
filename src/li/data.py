"""Prompt corpora for direction extraction and refusal evals.

Sources, all public:
- harmful:  AdvBench (520 harmful behaviors)
- harmless: Alpaca instruction subset (benign tasks)
- over-refusal probe: XSTest (exaggerated-safety prompts)
- perplexity: WikiText-2 test split

Each source is pinned to a revision: the splits below are fixed slices of a
seeded shuffle, so any upstream change would move prompts between them.
AdvBench and XSTest come from their authors' repos rather than the gated HF
mirrors, so results don't depend on whether the user has an HF token.
"""
from __future__ import annotations

import random

ADV_BENCH_CSV = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
    "098262edf85f807224e70ecd87b9d83716bf6b73/data/advbench/harmful_behaviors.csv"
)
HARMLESS_HF = "tatsu-lab/alpaca"
HARMLESS_REVISION = "dce01c9b08f87459cf36a430d809084718273017"
WIKITEXT_HF = "Salesforce/wikitext"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
XSTEST_CSV = (
    "https://raw.githubusercontent.com/paul-rottger/xstest/"
    "d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d/xstest_prompts.csv"
)

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
    rows = load_dataset("csv", data_files=ADV_BENCH_CSV, split="train")
    return _sample(list(rows["goal"]), n, seed, split)


def load_harmless(n: int = 128, seed: int = 0,
                  split: str | None = None) -> list[str]:
    from datasets import load_dataset
    rows = load_dataset(HARMLESS_HF, split="train", revision=HARMLESS_REVISION)
    return _sample([r["instruction"] for r in rows], n, seed, split)


def load_xstest(n: int = 100, seed: int = 0) -> list[str]:
    """XSTest's 250 safe prompts; the 200 unsafe contrast prompts should be
    refused, so including them would inflate the over-refusal rate."""
    from datasets import load_dataset
    rows = load_dataset("csv", data_files=XSTEST_CSV, split="train")
    return _sample([r["prompt"] for r in rows if r["label"] == "safe"], n, seed)


def load_wikitext() -> list[str]:
    """Non-empty lines of the wikitext-2-raw-v1 test split (perplexity)."""
    from datasets import load_dataset
    rows = load_dataset(WIKITEXT_HF, "wikitext-2-raw-v1", split="test",
                        revision=WIKITEXT_REVISION)
    return [t for t in rows["text"] if t.strip()]


def revisions() -> dict[str, str]:
    """Every pinned prompt source, for run provenance."""
    return {"advbench": ADV_BENCH_CSV, "alpaca": f"{HARMLESS_HF}@{HARMLESS_REVISION}",
            "xstest": XSTEST_CSV, "wikitext": f"{WIKITEXT_HF}@{WIKITEXT_REVISION}"}
