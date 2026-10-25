"""Evaluation suite: refusal, over-refusal, perplexity — one model dir at a time."""
from __future__ import annotations

import gc

import mlx.core as mx

from li.config import Config
from li.data import load_advbench, load_xstest
from li.evals.capability import perplexity
from li.evals.refusal import refusal_rate


def run_eval_suite(dirs: dict[str, str], cfg: Config) -> dict[str, dict[str, float]]:
    """Score every label->model_dir on refusal, over-refusal and perplexity."""
    advbench = load_advbench(cfg.n_advbench_eval, cfg.seed)
    xstest = load_xstest(cfg.n_xstest_eval, cfg.seed)
    results: dict[str, dict[str, float]] = {}
    from li.stream import default_budget_gb, load_for_inference, sequence_state_gb
    for label, model_dir in dirs.items():
        # leave room for one cache per batched sequence
        budget = cfg.stream_budget_gb or max(
            1.0, default_budget_gb() - cfg.eval_batch * sequence_state_gb(
                model_dir, cfg.max_seq + cfg.max_new_tokens))
        model, tokenizer, engine = load_for_inference(model_dir, budget)
        if engine:
            print(f"{label}: streaming — {engine.describe()}")
        results[label] = {
            "advbench_refusal": refusal_rate(model, tokenizer, advbench, cfg),
            "xstest_overrefusal": refusal_rate(model, tokenizer, xstest, cfg),
            "perplexity": perplexity(model, tokenizer, cfg),
        }
        del model, tokenizer, engine
        gc.collect()
        mx.clear_cache()
    return results
