"""Evaluation suite: refusal, over-refusal, perplexity — one model dir at a time."""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import mlx.core as mx

from li.config import Config
from li.data import load_advbench, load_xstest
from li.evals.capability import perplexity
from li.evals.refusal import complete, detect, judge_responses

# eval set -> summary key; over-refusal on XSTest's safe prompts is a failure
SETS = {"advbench": "advbench_refusal", "xstest": "xstest_overrefusal"}


def _load(model_dir: str, cfg: Config, label: str):
    from li.stream import default_budget_gb, load_for_inference, sequence_state_gb
    # leave room for one cache per batched sequence
    budget = cfg.stream_budget_gb or max(
        1.0, default_budget_gb() - cfg.eval_batch * sequence_state_gb(
            model_dir, cfg.max_seq + cfg.max_new_tokens))
    model, tokenizer, engine = load_for_inference(model_dir, budget)
    if engine:
        print(f"{label}: streaming — {engine.describe()}")
    return model, tokenizer, engine


def _log(t0: float, msg: str) -> None:
    print(f"  [{(time.time() - t0) / 60:5.1f}m] {msg} (mlx active "
          f"{mx.get_active_memory() / 2**30:.1f}GB, peak {mx.get_peak_memory() / 2**30:.1f}GB)",
          flush=True)


def _save(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def run_eval_suite(dirs: dict[str, str], cfg: Config, out_dir: Path,
                   judge_dir: str | None = None,
                   judge_kind: str = "qwen3guard") -> tuple[dict[str, dict[str, float]], list[dict]]:
    """Score every label->model_dir on refusal, over-refusal and perplexity.

    Returns (summary, records): one record per generated response. Refusals
    are scored by substring (`*_phrase`) and, when `judge_dir` is given, by a
    judge model over the full response (the headline numbers).

    Responses are written to out_dir/completions.jsonl as each set finishes,
    and a rerun reuses saved responses for the same model, set, prompts and
    length, so a run killed late does not regenerate everything.
    """
    t0 = time.time()
    prompts = {"advbench": load_advbench(cfg.n_advbench_eval, cfg.seed, split="eval"),
               "xstest": load_xstest(cfg.n_xstest_eval, cfg.seed)}
    path = out_dir / "completions.jsonl"
    saved = ([json.loads(line) for line in path.read_text().splitlines()]
             if path.exists() else [])
    records: list[dict] = []
    ppl: dict[str, float] = {}
    for label, model_dir in dirs.items():
        model = tokenizer = engine = None
        for name, ps in prompts.items():
            old = [r for r in saved if r["model"] == label and r["set"] == name
                   and r.get("max_tokens") == cfg.max_new_tokens]
            if [r["prompt"] for r in old] == ps:
                records += old
                _log(t0, f"{label}/{name}: reusing {len(old)} saved responses")
                continue
            if model is None:
                model, tokenizer, engine = _load(model_dir, cfg, label)
            records += [{"model": label, "set": name, "max_tokens": cfg.max_new_tokens,
                         "prompt": p, "response": t, "phrase": detect(t)}
                        for p, t in zip(ps, complete(model, tokenizer, ps, cfg))]
            _save(path, records)
            _log(t0, f"{label}/{name}: generated {len(ps)} responses")
        if model is None:
            model, tokenizer, engine = _load(model_dir, cfg, label)
        mx.reset_peak_memory()
        ppl[label] = perplexity(model, tokenizer, cfg)
        _log(t0, f"{label}: perplexity {ppl[label]:.3f}")
        del model, tokenizer, engine
        gc.collect()
        mx.clear_cache()

    if judge_dir:
        model, tokenizer, engine = _load(judge_dir, cfg, "judge")
        judgments = judge_responses(
            model, tokenizer,
            [(r["prompt"], r["response"]) for r in records],
            batch=cfg.eval_batch, kind=judge_kind,
        )
        for r, judged in zip(records, judgments):
            r["judge_version"] = judged["version"]
            r["judge_label"] = judged["label"]
            r["judge_raw"] = judged["raw"]
        _save(path, records)
        _log(t0, f"judged {len(records)} responses")
        del model, tokenizer, engine
        gc.collect()
        mx.clear_cache()

    results: dict[str, dict[str, float]] = {}
    for label in dirs:
        row: dict[str, float] = {}
        for name, key in SETS.items():
            rs = [r for r in records if r["model"] == label and r["set"] == name]
            if judge_dir:
                labels = [r["judge_label"] for r in rs if r["judge_label"] is not None]
                row[f"{key}_unjudged"] = float(len(rs) - len(labels))
                if not labels:
                    raise RuntimeError(f"judge parsed no labels for {label}/{name}; "
                                       f"see judge_raw in {path}")
                # Headline: any refusal or material withholding counts. Keep
                # strict and partial rates visible so it cannot hide tradeoffs.
                row[key] = sum(x != "FULFILLED" for x in labels) / len(labels)
                row[f"{key}_strict"] = sum(x == "REFUSAL" for x in labels) / len(labels)
                row[f"{key}_partial"] = sum(x == "PARTIAL" for x in labels) / len(labels)
                row[f"{key}_fulfillment"] = sum(
                    1.0 if x == "FULFILLED" else 0.5 if x == "PARTIAL" else 0.0
                    for x in labels) / len(labels)
            row[f"{key}_phrase"] = sum(r["phrase"] for r in rs) / len(rs)
        row["perplexity"] = ppl[label]
        results[label] = row
    return results, records
