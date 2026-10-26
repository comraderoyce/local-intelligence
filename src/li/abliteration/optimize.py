"""Heretic-style abliteration search: direction and per-layer strength by TPE.

A single direction at one fixed strength forces a bad trade on Qwen3.8-27B:
the layer-37 direction left 11/16 refusals (ppl 11.6 -> 13.6) while layer 34
removed all of them but nearly quadrupled perplexity (45.6). Following
p-e-w's Heretic, the edit gets a strength kernel over depth — peaking at one
layer and falling off linearly to a floor at `width` layers away, zero beyond,
separately scaled for attention and MLP writers — and the direction is
interpolated between adjacent layers at one template position. Optuna's TPE
sampler tunes those parameters against

    refusal fraction (short generations) + kl_weight * first-token KL on
    harmless prompts + steep penalties for KL past `kl_soft` and perplexity
    past `ppl_soft`

The penalties must stay gentle below the damage line: with a 10x KL penalty
from 0.15, an untouched model (100% refusals, KL 0) outscored a real partial
edit (69% refusals, KL 0.23, ppl +8%) and the search drifted to no-ops. On
the 27B, damage showed from KL ~0.3 (ppl x4) — hence the defaults.

Trials run on the live model with the edit applied as output hooks:
y' = y - a d (d.y) on each residual writer equals orthogonalizing its weight
(d.(Wx) = (W^T d).x), so no weights are touched until the winner is written
as a regular adapter (per-module lora_b = -a d).
"""
from __future__ import annotations

import json
import gc
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from safetensors.numpy import safe_open as np_safe_open

from ..config import Config
from ..data import load_advbench, load_harmless
from ..generate import chat_inputs

COMPONENTS = {
    "attn": ("self_attn.o_proj", "linear_attn.out_proj"),
    "mlp": ("mlp.down_proj", "mlp.shared_expert.down_proj"),
}
PPL_TOKENS = 256


class OutputAblation(nn.Module):
    """y - a d (d.y) on a residual writer's output. Holds the original as
    `.linear` so streamed weight names still map to the files."""

    def __init__(self, linear: nn.Module):
        super().__init__()
        self.linear = linear
        self.set(None, 0.0)

    def set(self, d: mx.array | None, a: float) -> None:
        object.__setattr__(self, "_d", d)
        object.__setattr__(self, "_a", float(a))

    def __call__(self, x):
        y = self.linear(x)
        if self._d is None or self._a == 0.0:
            return y
        return y - (self._a * ((y.astype(mx.float32) @ self._d.T) @ self._d)).astype(y.dtype)


class EmbedAblation(nn.Module):
    def __init__(self, embedding: nn.Module):
        super().__init__()
        self.embedding = embedding
        self.set(None, 0.0)

    set = OutputAblation.set

    def __call__(self, x):
        y = self.embedding(x)
        if self._d is None or self._a == 0.0:
            return y
        return y - (self._a * ((y.astype(mx.float32) @ self._d.T) @ self._d)).astype(y.dtype)

    def as_linear(self, x):  # tied output head: left as is
        return self.embedding.as_linear(x)


class Hooks:
    """Wraps every residual writer and the embedding of a (lazily loaded,
    not yet streamed) model; `apply` sets direction and per-layer strengths."""

    def __init__(self, model):
        from ..stream import _layers_path
        layers, _ = _layers_path(model)
        self.n_layers = len(layers)
        self.writers: list[tuple[int, str, OutputAblation]] = []
        for i, layer in enumerate(layers):
            for kind, paths in COMPONENTS.items():
                for path in paths:
                    *parents, attr = path.split(".")
                    parent = layer
                    for name in parents:
                        parent = dict.get(parent, name) if parent is not None else None
                    child = dict.get(parent, attr) if parent is not None else None
                    if child is not None:
                        wrapper = OutputAblation(child)
                        setattr(parent, attr, wrapper)
                        self.writers.append((i, kind, wrapper))
        inner = getattr(model, "language_model", model).model
        self.embed = EmbedAblation(inner.embed_tokens)
        inner.embed_tokens = self.embed

    def apply(self, d: mx.array | None, alphas: dict[str, list[float]], embed: float) -> None:
        for i, kind, w in self.writers:
            w.set(d if d is None or d.ndim == 2 else d[i], alphas[kind][i])
        self.embed.set(d if d is None or d.ndim == 2 else d[0], embed)


def kernel(n: int, peak: float, width: float, a_max: float, a_min: float) -> list[float]:
    """Strength per layer: a_max at `peak`, linear to a_min at `width` away, 0 beyond."""
    out = []
    for i in range(n):
        dist = abs(i - peak)
        out.append(a_max + (a_min - a_max) * dist / width if dist <= width else 0.0)
    return out


def _unit_projected(direction: np.ndarray, harmless_mean: np.ndarray | None) -> np.ndarray:
    v = direction.astype(np.float32)
    if harmless_mean is not None:
        m = harmless_mean.astype(np.float32)
        m /= max(float(np.linalg.norm(m)), 1e-12)
        v = v - float(v @ m) * m
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _directions(dirs_pos: np.ndarray, harmless_pos: np.ndarray, layer: float,
                pos_idx: int, scope: str, project_harmless: bool) -> np.ndarray:
    """Shared [1,D] interpolated direction or one [L,1,D] per layer."""
    harmless = harmless_pos if project_harmless else None
    if scope == "per_layer":
        rows = [_unit_projected(dirs_pos[i, pos_idx],
                                None if harmless is None else harmless[i, pos_idx])
                for i in range(dirs_pos.shape[0])]
        return np.asarray(rows, dtype=np.float32)[:, None, :]
    lo = int(math.floor(layer))
    hi = min(lo + 1, dirs_pos.shape[0] - 1)
    frac = layer - lo
    v = (1 - frac) * dirs_pos[lo, pos_idx] + frac * dirs_pos[hi, pos_idx]
    m = None
    if harmless is not None:
        m = (1 - frac) * harmless[lo, pos_idx] + frac * harmless[hi, pos_idx]
    return _unit_projected(v, m)[None]


def schedule(params: dict, n_layers: int) -> tuple[dict[str, list[float]], float]:
    alphas = {kind: kernel(n_layers, params["peak"], params["width"], params[f"{kind}_max"],
                           params[f"{kind}_max"] * params[f"{kind}_min_frac"])
              for kind in COMPONENTS}
    return alphas, params["embed"]


def optimize(model_dir: str, stats_path: str, cfg: Config, out_dir: str, out_report: str,
             n_trials: int = 40, n_prompts: int = 16, max_tokens: int = 64,
             batch: int = 16, kl_soft: float = 0.3, kl_weight: float = 0.5,
             ppl_soft: float = 1.10, budget_gb: float | None = None,
             warm_starts: list[dict] | None = None, storage: str | None = None,
             study_name: str = "abliterate", rerank: int = 5,
             rerank_tokens: int = 128, judge_dir: str | None = None,
             judge_kind: str = "qwen3guard") -> dict:
    import optuna
    from mlx_lm import batch_generate

    from ..evals.refusal import detect_withholding, judge_responses
    from ..stream import load_streamed
    from .apply import write_abliteration
    from .select import _first_token_logprobs

    with np_safe_open(stats_path, framework="numpy") as f:
        dirs_pos = f.get_tensor("directions_pos").astype(np.float32)
        harmless_pos = f.get_tensor("harmless_mean_pos").astype(np.float32)
    n_layers, n_pos, _ = dirs_pos.shape

    holder: dict = {}
    model, tokenizer, engine = load_streamed(
        model_dir, budget_gb, prepare=lambda m: holder.setdefault("hooks", Hooks(m)))
    hooks: Hooks = holder["hooks"]
    print(f"optimize: {engine.describe()}")

    # Fixed slice held out from measurement, selection, and final evaluation.
    harmful_texts = load_advbench(n_prompts, cfg.seed, split="optimize")
    harmful = [chat_inputs(tokenizer, p) for p in harmful_texts]
    harmless = load_harmless(n_prompts, cfg.seed, split="optimize")
    from datasets import load_dataset
    text = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                                              split="test")["text"] if t.strip())
    ppl_ids = mx.array([tokenizer.encode(text[:20000])[:PPL_TOKENS]])

    def short_ppl() -> float:
        lp = model(ppl_ids)[0, :-1].astype(mx.float32)
        lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
        tgt = ppl_ids[0, 1:]
        return math.exp(-lp[mx.arange(tgt.size), tgt].mean().item())

    def responses(tokens: int = max_tokens) -> list[str]:
        return batch_generate(model, tokenizer, harmful, max_tokens=tokens,
                               completion_batch_size=batch, prefill_batch_size=batch).texts

    def refusals() -> float:
        return sum(map(detect_withholding, responses())) / len(harmful)

    def normalized(params: dict) -> dict:
        # Old studies remain readable after adding the new search dimensions.
        return {"direction_scope": "global", "project_harmless": False, **params}

    hooks.apply(None, {k: [0.0] * n_layers for k in COMPONENTS}, 0.0)
    base_lp = _first_token_logprobs(model, tokenizer, harmless, None)
    base_ppl = short_ppl()
    base_ref = refusals()
    print(f"  base: refusals {base_ref:.0%}, ppl {base_ppl:.2f}", flush=True)

    def evaluate(params: dict) -> dict:
        params = normalized(params)
        d = mx.array(_directions(
            dirs_pos, harmless_pos, params["dir_layer"], params["dir_pos"] + n_pos,
            params["direction_scope"], params["project_harmless"],
        ))
        alphas, embed = schedule(params, n_layers)
        hooks.apply(d, alphas, embed)
        lp = _first_token_logprobs(model, tokenizer, harmless, None)
        kl = mx.mean(mx.sum(mx.exp(base_lp) * (base_lp - lp), axis=-1)).item()
        ppl = short_ppl()
        ref = refusals()
        score = (ref + kl_weight * kl + 3 * max(0.0, kl - kl_soft)
                 + 5 * max(0.0, ppl / base_ppl - ppl_soft))
        return {"refusals": ref, "kl": kl, "ppl": ppl, "score": score}

    def objective(trial: optuna.Trial) -> float:
        params = {
            "direction_scope": trial.suggest_categorical(
                "direction_scope", ["global", "per_layer"]),
            "project_harmless": trial.suggest_categorical(
                "project_harmless", [False, True]),
            "dir_layer": trial.suggest_float("dir_layer", 0.2 * n_layers, 0.8 * (n_layers - 1)),
            "dir_pos": trial.suggest_int("dir_pos", -n_pos, -1),
            "peak": trial.suggest_float("peak", 0, n_layers - 1),
            "width": trial.suggest_float("width", 2, n_layers),
            # floors above 0: near-zero strengths are no-ops the search wastes trials on
            "attn_max": trial.suggest_float("attn_max", 0.3, 1.5),
            "attn_min_frac": trial.suggest_float("attn_min_frac", 0.0, 1.0),
            "mlp_max": trial.suggest_float("mlp_max", 0.3, 1.5),
            "mlp_min_frac": trial.suggest_float("mlp_min_frac", 0.0, 1.0),
            "embed": trial.suggest_float("embed", 0.0, 1.0),
        }
        t = time.time()
        m = evaluate(params)
        for k, v in m.items():
            trial.set_user_attr(k, v)
        print(f"  trial {trial.number:3d}: refusals {m['refusals']:.0%} kl {m['kl']:.3f} "
              f"ppl {m['ppl']:.2f} score {m['score']:.3f}  ({time.time() - t:.0f}s)",
              flush=True)
        return m["score"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", study_name=study_name,
                                storage=storage, load_if_exists=True)
    # reseed per resume: a fixed seed replays the same random startup trials
    study.sampler = optuna.samplers.TPESampler(seed=cfg.seed + len(study.trials),
                                               n_startup_trials=10)
    for ws in (warm_starts or []) if not study.trials else []:
        study.enqueue_trial(ws)
    remaining = n_trials - len([t for t in study.trials if t.state.is_finished()])
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)

    optuna_best = study.best_trial
    chosen = optuna_best
    semantic: dict[int, dict] = {}
    candidate_texts: dict[int, list[str]] = {}
    if rerank > 0 and judge_dir:
        candidates = sorted(
            (t for t in study.trials if t.state.is_finished() and "score" in t.user_attrs),
            key=lambda t: t.user_attrs["score"],
        )[:rerank]
        zero = {k: [0.0] * n_layers for k in COMPONENTS}
        for trial in candidates:
            params = normalized(dict(trial.params))
            d_live = mx.array(_directions(
                dirs_pos, harmless_pos, params["dir_layer"], params["dir_pos"] + n_pos,
                params["direction_scope"], params["project_harmless"],
            ))
            alphas_live, embed_live = schedule(params, n_layers)
            hooks.apply(d_live, alphas_live, embed_live)
            candidate_texts[trial.number] = responses(rerank_tokens)
        hooks.apply(None, zero, 0.0)
        # Free the edited model before loading the independent classifier.
        del model, tokenizer, engine
        gc.collect()
        mx.clear_cache()
        from ..stream import load_for_inference
        judge, judge_tokenizer, judge_engine = load_for_inference(judge_dir)
        for trial in candidates:
            judged = judge_responses(
                judge, judge_tokenizer,
                list(zip(harmful_texts, candidate_texts[trial.number])),
                batch=batch, kind=judge_kind,
            )
            labels = [j["label"] for j in judged]
            refusal = sum(x != "FULFILLED" for x in labels) / len(labels)
            strict = sum(x == "REFUSAL" for x in labels) / len(labels)
            partial = sum(x == "PARTIAL" for x in labels) / len(labels)
            kl, ppl = trial.user_attrs["kl"], trial.user_attrs["ppl"]
            score = (refusal + kl_weight * kl + 3 * max(0.0, kl - kl_soft)
                     + 5 * max(0.0, ppl / base_ppl - ppl_soft))
            semantic[trial.number] = {
                "refusals": refusal, "strict_refusals": strict,
                "partial": partial, "score": score, "labels": labels,
            }
            print(f"  rerank {trial.number:3d}: refusals {refusal:.0%} "
                  f"(strict {strict:.0%}, partial {partial:.0%}) score {score:.3f}",
                  flush=True)
        if semantic:
            chosen = min(candidates, key=lambda t: semantic[t.number]["score"])
        del judge, judge_tokenizer, judge_engine
        gc.collect()
        mx.clear_cache()
    else:
        if rerank > 0:
            print("  semantic rerank skipped: no dedicated judge configured", flush=True)
        del model, tokenizer, engine
        gc.collect()
        mx.clear_cache()

    params = normalized(dict(chosen.params))
    d = _directions(
        dirs_pos, harmless_pos, params["dir_layer"], params["dir_pos"] + n_pos,
        params["direction_scope"], params["project_harmless"],
    )
    alphas, embed = schedule(params, n_layers)
    def alpha_of(layer: int | None, kind: str) -> float:
        return embed if kind == "embed" else alphas[kind][layer]

    chosen_metrics = {**chosen.user_attrs, **semantic.get(chosen.number, {})}
    meta = {"method": "optimized", "params": params, "metrics": chosen_metrics,
            "alphas": alphas, "embed": embed}
    result = write_abliteration(model_dir, d, out_dir, alpha_of, meta)
    report = {"base": {"refusals": base_ref, "ppl": base_ppl},
        "optuna_best": {"number": optuna_best.number,
                        "params": normalized(dict(optuna_best.params)),
                        **optuna_best.user_attrs},
        "best": {"number": chosen.number, "params": params,
                 **chosen.user_attrs, **semantic.get(chosen.number, {})},
        "semantic_rerank": semantic,
        "trials": [{"number": t.number, "params": t.params, **t.user_attrs}
                   for t in study.trials if t.state.is_finished()]}
    Path(out_report).write_text(json.dumps(report, indent=2) + "\n")
    return {**report, "adapter": result}
