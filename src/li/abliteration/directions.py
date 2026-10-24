"""Refusal-direction measurement for abliteration.

Captures last-position residual-stream hidden states for harmful and
harmless prompt sets with one batch-1 forward pass per prompt (bypassing
mlx_lm.generate), then derives a per-layer direction
d_l = normalize(mean(harmful) - mean(harmless)) and a selection score
(mlabonne): the mean cosine between centered harmful captures and d_l.
"""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask
from safetensors.numpy import save_file

from ..config import Config
from ..data import load_advbench, load_harmless
from ..generate import chat_inputs


def _capture_last_positions(model, tokenizer, prompts: list[str]) -> np.ndarray:
    """Return [P, L, D] float32: residual stream at the last token after
    every decoder layer, one batch-1 forward pass per prompt."""
    layers = model.model.layers
    n_layers = len(layers)
    out: np.ndarray | None = None
    for pi, prompt in enumerate(prompts):
        ids = chat_inputs(tokenizer, prompt)
        h = model.model.embed_tokens(mx.array([ids]))
        if out is None:
            out = np.zeros((len(prompts), n_layers, h.shape[-1]), dtype=np.float32)
        mask = create_attention_mask(h, None)
        for li, layer in enumerate(layers):
            h = layer(h, mask)
            out[pi, li] = np.array(h[0, -1].astype(mx.float32))
        mx.clear_cache()
    return out


def _layer_scores(harmful: np.ndarray, mu_harmless: np.ndarray,
                  direction: np.ndarray) -> np.ndarray:
    """Per-layer selection score: mean over harmful prompts of
    cosine(h_i - mu_harmless, d). `direction` must be L2-normalized."""
    n_layers = direction.shape[0]
    scores = np.zeros(n_layers, dtype=np.float32)
    for l in range(n_layers):
        centered = harmful[:, l, :] - mu_harmless[l][None, :]
        norms = np.maximum(np.linalg.norm(centered, axis=1), 1e-12)
        scores[l] = (centered @ direction[l] / norms).mean()
    return scores


def measure_directions(model_dir: str, cfg: Config,
                       out_stats: str, out_report: str) -> dict:
    model, tokenizer = load(model_dir)
    harmful = load_advbench(cfg.n_harmful, cfg.seed)
    harmless = load_harmless(cfg.n_harmless, cfg.seed)

    h_harm = _capture_last_positions(model, tokenizer, harmful)
    h_ben = _capture_last_positions(model, tokenizer, harmless)
    del model, tokenizer
    mx.clear_cache()

    mu_harm = h_harm.mean(axis=0)          # [L, D]
    mu_ben = h_ben.mean(axis=0)            # [L, D]
    d = mu_harm - mu_ben                   # [L, D]
    norms = np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    directions = (d / norms).astype(np.float32)

    scores = _layer_scores(h_harm, mu_ben, directions)
    # The separation score peaks in early layers on small models, but abliteration
    # from early layers destroys the model (verified on Qwen3-0.6B: coherent text at
    # layer 15, mojibake at layer 1 with identical code). Restrict auto-selection to
    # the upper 2/3 of the network; explicit --layer overrides this.
    cutoff = directions.shape[0] // 3
    best_layer = cutoff + int(np.argmax(scores[cutoff:]))

    for parent in (Path(out_stats).parent, Path(out_report).parent):
        parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "directions": directions,
            "harmful_mean": mu_harm.astype(np.float32),
            "harmless_mean": mu_ben.astype(np.float32),
            "scores": scores,
        },
        out_stats,
    )

    report = {
        "model_dir": model_dir,
        "n_harmful": len(harmful),
        "n_harmless": len(harmless),
        "n_layers": int(directions.shape[0]),
        "best_layer": best_layer,
        "scores": [float(s) for s in scores],
    }
    Path(out_report).write_text(json.dumps(report, indent=2) + "\n")
    return report
