# local-intelligence (`li`)

Abliteration + LoRA fine-tuning harness for open models on Apple Silicon.
Designed for 8GB and 16GB Macs, generalizes upward (32GB+).

## What it does

1. **download** — pulls a pre-quantized MLX model (4-bit for ≤14B, 3-bit for 30B+)
2. **measure** — streams harmful (AdvBench) vs harmless (Alpaca) prompts through the
   model one at a time, captures the chat-template suffix at every layer, and computes
   per-layer/per-position diff-in-means refusal directions
3. **abliterate** — orthogonalizes the chosen direction out of the weight matrices that
   write to the residual stream (`o_proj`, `down_proj`, per-expert for MoE) at the chosen
   layer and above. Quantized bases get an exact low-rank correction adapter, avoiding
   the loss caused by requantizing a small weight edit
4. **optimize** — searches the direction, harmless projection, global vs per-layer
   scope, and attention/MLP depth schedules; finalists are reranked on longer responses
5. **eval** — dedicated-judge refusal rate (AdvBench), over-refusal rate (XSTest),
   phrase-match diagnostics, and perplexity (wikitext-2) side by side
6. **train** — LoRA SFT via `mlx_lm.lora` on your task data
7. **fuse** — merges the adapter into the weights; serve with `mlx_lm.server` or `li chat`

Ordering matters: **abliterate → train**. Training after abliteration keeps the refusal
removal; abliterating after training can erase learned behavior.

## Setup

```sh
uv sync
li init                 # interactive: model registry + your RAM → li.toml
li download
```

## Usage

```sh
li measure                          # ranks layers, writes runs/<model>/report.json
li abliterate                       # uses best layer; --layer N --alpha 0.8 to tune
li optimize --trials 100            # searched adapter; resumable Optuna study
li eval                             # base vs abliterated, with configured judge
li chat                             # probe the abliterated model manually
li train --data task.jsonl --iters 1000
li fuse                             # merged model in runs/<model>/fused
```

`examples/sample_task.jsonl` shows the training data format (instruction/output JSONL;
`prompt`/`completion` and chat `messages` schemas also accepted).

`alpha` is the knob: 1.0 = full direction removal, lower = gentler. If eval shows high
refusal still, raise alpha or pick a different layer; if perplexity jumps or the model
gets incoherent, lower it.

Measurement, selection, optimization, and final evaluation use fixed non-overlapping
prompt slices. The cheap phrase heuristic is diagnostic only. For headline refusal
numbers, configure a purpose-trained local judge such as Qwen3Guard-Gen and set
`judge_dir`/`judge_kind` in `li.toml`; the model under test should not grade itself.

Layer auto-selection ignores the top third of the network: the separation score peaks
in early layers on small models, but abliterating there destroys the model (verified:
Qwen3-0.6B answers cleanly when ablated at layer 16/28, produces mojibake at layer 1).
`--layer N` overrides.

## Which model on which machine

| model | quant | abliterate | local LoRA |
|---|---|---|---|
| Qwen3-4B | 4-bit | 8GB ✓ | 8GB ✓ |
| Qwen3-8B | 4-bit | 8GB ✓ (tight) | 16GB ✓ |
| Qwen3-14B | 4-bit | 16GB ✓ | 32GB ✓ |
| Qwen3-30B-A3B (MoE) | 3-bit | 16GB ✓ | 64GB+ |
| Qwen3.8-27B (hybrid) | 3-bit | 16GB ✓ (eval/chat need 24GB) | 64GB+ |
| Qwen3-32B | 3-bit | 16GB ✓ | 64GB+ |

`li init` checks your RAM and refuses (or with `--force`, warns) tiers that can't run.
Training tiers are stricter than abliteration: inference + weight edits need only
weights + activations; LoRA needs adapter + optimizer state on top.

Any HF model id also works (`li init --model org/name`) — it converts from fp16 locally,
which needs disk but not RAM (streamed conversion).

## Notes

- Qwen3.8-27B is a hybrid model (Gated DeltaNet `linear_attn` layers plus a
  full-attention layer every 4th) in a multimodal-wrapped checkpoint
  (`language_model.model.layers...`). Abliteration edits `linear_attn.out_proj`,
  `self_attn.o_proj` and `mlp.down_proj`. There is no mlx-community 3-bit build, so the
  registry pulls a text-only community quant (`lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly`).
- Full-model inference (`eval`, `chat`) is refused when weights exceed 60% of RAM
  (`--force` overrides). Loading the 11GB 3-bit 27B on a 16GB Mac starved WindowServer
  into a watchdog kernel panic, worse with `iogpu.wired_limit_mb` raised — don't raise it.
  `measure` loads the model lazily and runs one layer at a time over all prompts, and
  `abliterate` rewrites shards in ~1GB parts, so both stay at a few GB on any size model.
- Thinking mode (Qwen3) is disabled for all harness generation for speed and
  consistency; the final fused model still supports it in normal use.
- Small models (≤4B) lose more general capability from abliteration than 8B+. Always
  check the eval table's perplexity delta before shipping a model into use.
- `li.toml` is machine-local (gitignored): same repo, different Macs, different configs.
