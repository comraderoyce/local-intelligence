"""Capability eval: perplexity on wikitext-2-raw-v1 (test split)."""
from __future__ import annotations

import math

import mlx.core as mx

from li.config import Config

MIN_WINDOWS = 3


def perplexity(model, tokenizer, cfg: Config) -> float:
    """exp(mean NLL) over predicted positions of consecutive max_seq windows."""
    from datasets import load_dataset

    texts = [
        ex["text"]
        for ex in load_dataset(
            "Salesforce/wikitext", "wikitext-2-raw-v1", split="test"
        )
        if ex["text"].strip()
    ]
    ids = tokenizer.encode("\n\n".join(texts))
    need = MIN_WINDOWS * cfg.max_seq
    if len(ids) < need:
        raise ValueError(
            f"wikitext test yields {len(ids)} tokens < {need} needed for "
            f"{MIN_WINDOWS} windows of {cfg.max_seq}"
        )
    ids = ids[:need]

    total_nll, predicted = 0.0, 0
    for start in range(0, need, cfg.max_seq):
        window = ids[start : start + cfg.max_seq]
        out = model(mx.array([window]))
        logits = getattr(out, "logits", out)[0, :-1, :].astype(mx.float32)
        # logprob of each actually-predicted token under the preceding context
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        targets = mx.array(window[1:])
        picked = logprobs[mx.arange(targets.shape[0]), targets]
        total_nll += -picked.sum().item()
        predicted += targets.shape[0]
        mx.clear_cache()
    return math.exp(total_nll / predicted)
