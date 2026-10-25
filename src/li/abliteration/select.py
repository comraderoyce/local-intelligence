"""Pick the refusal direction by intervention, not by separation score alone.

The separation score happily picks template-formatting positions (on
Qwen3-0.6B: the newline after <|im_end|>) whose directions turn the model
into token soup, which a substring refusal check then scores as 0% refusal.
Following Arditi et al., each candidate (layer, pos) direction is instead
ablated from the residual stream — removed from the embedding output and from
every layer's output, which is exactly what orthogonalizing all writers does —
and scored on one batched forward pass per prompt set:

- refusal score on harmful prompts: mean log-odds that the first response
  token is a refusal opener (lower = fewer refusals)
- KL divergence of the first-token distribution on harmless prompts vs the
  unablated model (higher = the edit damages ordinary behaviour)

The winner is the lowest refusal score with KL under `kl_max`. Arditi et al.
use 0.1; first-token KL on small models moves more easily (Qwen3-0.6B: no
candidate under 0.1 reduced refusal, layer 17 at 0.18 cut its score by 5.4).
"""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors.numpy import safe_open as np_safe_open

from ..config import Config
from ..data import load_advbench, load_harmless
from ..generate import chat_inputs

REFUSAL_OPENERS = ("I", "Sorry", "Unfortunately", "As an")
LAYER_RANGE = (0.2, 0.8)   # Arditi et al. exclude the last 20% of layers


def _refusal_token_ids(tokenizer) -> list[int]:
    ids = {tokenizer.encode(w, add_special_tokens=False)[0] for w in REFUSAL_OPENERS}
    return sorted(ids)


def _first_token_logprobs(model, tokenizer, prompts: list[str], dirs: mx.array | None):
    """[P, vocab] log-probs of the first response token, prompts right-padded
    into one batch (causal attention and recurrent state never see padding
    that comes after a position), with `dirs` ablated from the residual."""
    from ..stream import EngineLayer
    seqs = [chat_inputs(tokenizer, p) for p in prompts]
    n = max(map(len, seqs))
    pad = tokenizer.pad_token_id or 0
    ids = mx.array([s + [pad] * (n - len(s)) for s in seqs])
    last = mx.array([len(s) - 1 for s in seqs])
    lm = getattr(model, "language_model", model)
    inner = lm.model
    for layer in inner.layers:
        if isinstance(layer, EngineLayer):
            layer.set_ablation(dirs)
    h = inner(ids)                                     # final-normed hidden
    h = h[mx.arange(len(seqs)), last]                  # [P, D]
    head = getattr(lm, "lm_head", None)
    logits = head(h) if head is not None else inner.embed_tokens.as_linear(h)
    logits = logits.astype(mx.float32)
    out = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    mx.eval(out)
    return out


def _candidates(scores: np.ndarray, per_pos: int) -> list[tuple[int, int]]:
    """Top `per_pos` layers by separation score at each template position,
    within LAYER_RANGE of the depth."""
    n_layers, n_pos = scores.shape
    lo, hi = int(n_layers * LAYER_RANGE[0]), int(n_layers * LAYER_RANGE[1])
    out = []
    for i in range(n_pos):
        top = lo + np.argsort(-scores[lo:hi, i])[:per_pos]
        out += [(int(l), i) for l in sorted(top)]
    return out


def select_direction(model_dir: str, stats_path: str, cfg: Config, out_report: str,
                     n_prompts: int = 16, per_pos: int = 3, kl_max: float = 0.2,
                     budget_gb: float | None = None) -> dict:
    from ..stream import load_streamed
    with np_safe_open(stats_path, framework="numpy") as f:
        dirs_all = f.get_tensor("directions_pos").astype(np.float32)
        scores = f.get_tensor("scores_pos").astype(np.float32)
    n_pos = dirs_all.shape[1]
    model, tokenizer, engine = load_streamed(model_dir, budget_gb)
    print(f"selection: {engine.describe()}")
    # held out from measurement: a different seed's samples
    harmful = load_advbench(n_prompts, cfg.seed + 1)
    harmless = load_harmless(n_prompts, cfg.seed + 1)
    refusal_ids = mx.array(_refusal_token_ids(tokenizer))

    def refusal_score(lp: mx.array) -> float:
        p = mx.exp(lp[:, refusal_ids]).sum(axis=-1)
        return mx.mean(mx.log(p + 1e-8) - mx.log(1 - p + 1e-8)).item()

    base_ben = _first_token_logprobs(model, tokenizer, harmless, None)
    base_refusal = refusal_score(_first_token_logprobs(model, tokenizer, harmful, None))
    print(f"  base refusal score {base_refusal:+.2f}")
    rows = []
    for layer, i in _candidates(scores, per_pos):
        d = mx.array(dirs_all[layer, i][None])
        ref = refusal_score(_first_token_logprobs(model, tokenizer, harmful, d))
        lp = _first_token_logprobs(model, tokenizer, harmless, d)
        kl = mx.mean(mx.sum(mx.exp(base_ben) * (base_ben - lp), axis=-1)).item()
        rows.append({"layer": layer, "pos": i - n_pos, "refusal": ref, "kl": kl,
                     "score": float(scores[layer, i])})
        print(f"  layer {layer:3d} pos {i - n_pos:3d}  refusal {ref:+6.2f}  kl {kl:.3f}",
              flush=True)
    ok = [r for r in rows if r["kl"] < kl_max]
    best = min(ok or rows, key=lambda r: r["refusal"])
    report = {"base_refusal": base_refusal, "kl_max": kl_max, "best": best,
              "within_kl": bool(ok), "candidates": rows}
    Path(out_report).write_text(json.dumps(report, indent=2) + "\n")
    return report
