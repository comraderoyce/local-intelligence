"""Interactive chat and speed benchmarks over resident or streamed models.

Speculative decoding is used only for streamed models: there a forward pass
is dominated by reading weights, so verifying several draft tokens per pass
is nearly free. A resident model is compute-bound and gains little.
"""
from __future__ import annotations

import gc
import time

import mlx.core as mx

from .generate import chat_ids, chat_inputs
from .stream import load_for_inference, make_rewindable_cache, wrap_for_rewind


def _load(model_dir: str, draft_dir: str, budget_gb: float | None,
          num_draft: int, stream: bool | None = None):
    from mlx_lm import load
    model, tokenizer, engine = load_for_inference(model_dir, budget_gb, stream)
    draft = None
    if engine and draft_dir:
        draft, _ = load(draft_dir)
        wrap_for_rewind(draft)
    return model, tokenizer, engine, draft


def _new_cache(model, draft, num_draft: int) -> list:
    from mlx_lm.models.cache import make_prompt_cache
    if draft is None:
        return make_prompt_cache(model)
    # snapshots must cover a verify pass of num_draft + 1 tokens
    depth = num_draft + 3
    return make_rewindable_cache(model, depth) + make_rewindable_cache(draft, depth)


def _generate(model, tokenizer, ids, cache, draft, num_draft, max_tokens, echo):
    from mlx_lm import stream_generate
    kw = {"draft_model": draft, "num_draft_tokens": num_draft} if draft else {}
    t0, t1 = time.time(), None
    n = accepted = 0
    text = []
    for r in stream_generate(model, tokenizer, prompt=ids, max_tokens=max_tokens,
                             prompt_cache=cache, **kw):
        if t1 is None:
            t1 = time.time()
        n += 1
        accepted += bool(getattr(r, "from_draft", False))
        text.append(r.text)
        if echo:
            print(r.text, end="", flush=True)
    t2 = time.time()
    return {"prefill_s": (t1 or t2) - t0, "tokens": n, "text": "".join(text),
            "tok_s": (n - 1) / (t2 - t1) if n > 1 else 0.0,
            "accepted": accepted / n if n else 0.0}


def run_chat(model_dir: str, draft_dir: str, budget_gb: float | None,
             num_draft: int, max_tokens: int) -> None:
    model, tokenizer, engine, draft = _load(model_dir, draft_dir, budget_gb, num_draft)
    print(f"{model_dir}: " + (engine.describe() if engine else "fully resident")
          + (f", speculative with {draft_dir}" if draft else ""))
    print("type a message; 'q' quits, '/reset' clears history\n")
    # The conversation is re-templated every turn rather than appended to a
    # running cache: generation feeds the end-of-turn token into the cache but
    # not the separator after it, a reply cut at max_tokens has no end-of-turn
    # at all, and templates like Qwen3's rewrite earlier assistant turns.
    messages: list[dict] = []
    while True:
        try:
            user = input(">> ").strip()
        except EOFError:
            break
        if user == "q":
            break
        if user == "/reset":
            messages.clear()
            continue
        if not user:
            continue
        messages.append({"role": "user", "content": user})
        stats = _generate(model, tokenizer, chat_ids(tokenizer, messages),
                          _new_cache(model, draft, num_draft),
                          draft, num_draft, max_tokens, echo=True)
        messages.append({"role": "assistant", "content": stats["text"]})
        extra = f", {stats['accepted']:.0%} drafted" if draft else ""
        print(f"\n[{stats['tok_s']:.2f} tok/s, prefill {stats['prefill_s']:.1f}s{extra}]\n")


BENCH_PROMPT = "Explain in three sentences how a refrigerator works."


def run_bench(dirs: dict[str, str], draft_dir: str, budget_gb: float | None,
              num_draft: int, max_tokens: int, cfg) -> dict[str, dict]:
    """Speed (plain and speculative) and perplexity per model dir."""
    from .evals.capability import perplexity
    results = {}
    for label, model_dir in dirs.items():
        mx.reset_peak_memory()
        model, tokenizer, engine, draft = _load(model_dir, draft_dir, budget_gb, num_draft)
        ids = chat_inputs(tokenizer, BENCH_PROMPT)
        row = {
            "resident_gb": engine.resident_gb if engine else None,
            "streamed_gb": engine.streamed_gb if engine else 0.0,
        }
        plain = _generate(model, tokenizer, ids, _new_cache(model, None, 0),
                          None, 0, max_tokens, echo=False)
        row["plain_tok_s"] = plain["tok_s"]
        row["prefill_s"] = plain["prefill_s"]
        if draft:
            spec = _generate(model, tokenizer, ids, _new_cache(model, draft, num_draft),
                             draft, num_draft, max_tokens, echo=False)
            row["spec_tok_s"], row["spec_accepted"] = spec["tok_s"], spec["accepted"]
        row["perplexity"] = perplexity(model, tokenizer, cfg)
        row["peak_mlx_gb"] = mx.get_peak_memory() / 2**30
        results[label] = row
        print(f"{label}: {row}")
        del model, tokenizer, engine, draft
        gc.collect()
        mx.clear_cache()
    return results
