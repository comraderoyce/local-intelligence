"""Shared MLX-LLM helpers: chat templating and generation.

All harness text generation goes through here so prompt formatting is
identical between direction measurement, evals, and manual chat.
Thinking mode is disabled by default (Qwen3): shorter, cheaper outputs.
"""
from __future__ import annotations


def chat_inputs(tokenizer, user: str, system: str | None = None) -> list[int]:
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user}
    ]
    try:
        ids = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:  # non-Qwen3 template without thinking support
        ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    return ids


def chat_generate(model, tokenizer, user: str, system: str | None = None,
                  max_tokens: int = 256) -> str:
    from mlx_lm import generate
    ids = chat_inputs(tokenizer, user, system)
    return generate(model, tokenizer, prompt=ids, max_tokens=max_tokens, verbose=False)
