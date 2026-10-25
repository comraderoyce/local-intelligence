"""Refusal-direction measurement for abliteration.

Captures residual-stream hidden states for harmful and harmless prompt sets
with batch-1 forward passes (bypassing mlx_lm.generate) at every
post-instruction position: the chat-template tokens after the user message.
The last token alone is a weak choice when the template ends in formatting
(Qwen with thinking disabled ends in an empty `<think></think>` block), so,
as in Arditi et al., each (layer, position) gets a direction
d = normalize(mean(harmful) - mean(harmless)) and a selection score
(mlabonne): the mean cosine between centered harmful captures and d.
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

MAX_POSITIONS = 16


def template_suffix_len(tokenizer) -> int:
    """Number of chat-template tokens after the user message: the common token
    suffix of two prompts with different content."""
    a = chat_inputs(tokenizer, "alpha beta gamma")
    b = chat_inputs(tokenizer, "a different request entirely")
    n = 0
    while n < min(len(a), len(b)) and a[-1 - n] == b[-1 - n]:
        n += 1
    return max(1, min(n, MAX_POSITIONS))


def _stats(h_harm: np.ndarray, h_ben: np.ndarray):
    """Per-position direction, harmless mean and score for one layer.
    h_*: [P, S, D] float32."""
    mu_h, mu_b = h_harm.mean(axis=0), h_ben.mean(axis=0)            # [S, D]
    diff = mu_h - mu_b
    d = diff / np.maximum(np.linalg.norm(diff, axis=-1, keepdims=True), 1e-12)
    centered = h_harm - mu_b[None]                                    # [P, S, D]
    norms = np.maximum(np.linalg.norm(centered, axis=-1), 1e-12)
    scores = (np.einsum("psd,sd->ps", centered, d) / norms).mean(axis=0)
    return d, mu_b, scores


def _capture(model, tokenizer, harmful: list[str], harmless: list[str], n_pos: int):
    """Layer-major capture of the last `n_pos` positions after every layer.

    Layer-major so the model never has to be resident: with a lazily loaded
    model, each layer's weights are materialized once, run over every prompt,
    then dropped. Peak memory is one layer plus every prompt's hidden states,
    instead of the whole model (a 3-bit 27B on 16GB starves WindowServer
    into a watchdog panic otherwise). Statistics are reduced per layer, so
    nothing scales with layers x prompts x positions.
    """
    inner = model.model
    layers = inner.layers
    n_layers = len(layers)
    prompts = harmful + harmless
    hs = [inner.embed_tokens(mx.array([chat_inputs(tokenizer, p)])) for p in prompts]
    mx.eval(hs)
    inner.embed_tokens = None
    mx.clear_cache()
    dim = hs[0].shape[-1]
    directions = np.zeros((n_layers, n_pos, dim), dtype=np.float32)
    harmless_mean = np.zeros_like(directions)
    scores = np.zeros((n_layers, n_pos), dtype=np.float32)
    cap = np.zeros((len(prompts), n_pos, dim), dtype=np.float32)
    for li in range(n_layers):
        layer = layers[li]
        # hybrid models (Qwen3.5+): linear-attention layers take an SSM mask,
        # which is None for an unpadded batch-1 prompt
        linear = getattr(layer, "is_linear", False)
        for pi, h in enumerate(hs):
            h = layer(h, None if linear else create_attention_mask(h, None))
            mx.eval(h)
            hs[pi] = h
            cap[pi] = np.array(h[0, -n_pos:].astype(mx.float32))
        layers[li] = None
        del layer
        mx.clear_cache()
        directions[li], harmless_mean[li], scores[li] = _stats(
            cap[: len(harmful)], cap[len(harmful):])
        print(f"  layer {li + 1}/{n_layers}", end="\r", flush=True)
    print()
    return directions, harmless_mean, scores


def measure_directions(model_dir: str, cfg: Config,
                       out_stats: str, out_report: str) -> dict:
    model, tokenizer = load(model_dir, lazy=True)
    harmful = load_advbench(cfg.n_harmful, cfg.seed)
    harmless = load_harmless(cfg.n_harmless, cfg.seed)
    n_pos = template_suffix_len(tokenizer)

    # one pass over both sets: layers are consumed as they are captured
    directions, harmless_mean, scores = _capture(model, tokenizer, harmful, harmless, n_pos)
    del model, tokenizer
    mx.clear_cache()

    # The separation score peaks in early layers on small models, but abliteration
    # from early layers destroys the model (verified on Qwen3-0.6B: coherent text at
    # layer 15, mojibake at layer 1 with identical code). Restrict auto-selection to
    # the upper 2/3 of the network; explicit --layer overrides this.
    n_layers = directions.shape[0]
    cutoff = n_layers // 3
    flat = int(np.argmax(scores[cutoff:]))
    best_layer, best_idx = cutoff + flat // n_pos, flat % n_pos
    best_pos = best_idx - n_pos                      # negative offset from the end

    for parent in (Path(out_stats).parent, Path(out_report).parent):
        parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            # [L, D] at the chosen position, for callers that take one per layer
            "directions": np.ascontiguousarray(directions[:, best_idx]),
            "directions_pos": directions,             # [L, S, D], S = template suffix
            "harmless_mean_pos": harmless_mean,
            "scores_pos": scores,
            "scores": np.ascontiguousarray(scores[:, best_idx]),
        },
        out_stats,
    )

    report = {
        "model_dir": model_dir,
        "n_harmful": len(harmful),
        "n_harmless": len(harmless),
        "n_layers": n_layers,
        "n_positions": n_pos,
        "best_layer": best_layer,
        "best_pos": best_pos,
        "scores": [float(s) for s in scores[:, best_idx]],
        "scores_pos": [[float(s) for s in row] for row in scores],
    }
    Path(out_report).write_text(json.dumps(report, indent=2) + "\n")
    return report
