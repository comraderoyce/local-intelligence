# local-intelligence (`li`)

Abliteration + LoRA fine-tuning harness for open models on Apple Silicon.
Designed for 8GB and 16GB Macs, generalizes upward (32GB+).

## What it does

1. **download** — pulls a pre-quantized MLX model (4-bit for ≤14B, 3-bit for 30B+)
2. **measure** — streams harmful (AdvBench) vs harmless (Alpaca) prompts through the
   model one at a time, captures last-position residual activations at every layer,
   computes per-layer diff-in-means refusal directions and a separation score
3. **abliterate** — orthogonalizes the chosen direction out of the weight matrices that
   write to the residual stream (`o_proj`, `down_proj`, per-expert for MoE) at the chosen
   layer and above. Works **on the quantized weights**: dequantize tensor → edit →
   requantize, so peak RAM is a few tensors, never a full-precision model
4. **eval** — refusal rate (AdvBench), over-refusal rate (XSTest), perplexity
   (wikitext-2) for base vs abliterated side by side
5. **train** — LoRA SFT via `mlx_lm.lora` on your task data
6. **fuse** — merges the adapter into the weights; serve with `mlx_lm.server` or `li chat`

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
li eval                             # base vs abliterated table
li chat                             # probe the abliterated model manually
li train --data task.jsonl --iters 1000
li fuse                             # merged model in runs/<model>/fused
```

`examples/sample_task.jsonl` shows the training data format (instruction/output JSONL;
`prompt`/`completion` and chat `messages` schemas also accepted).

`alpha` is the knob: 1.0 = full direction removal, lower = gentler. If eval shows high
refusal still, raise alpha or pick a different layer; if perplexity jumps or the model
gets incoherent, lower it.

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
| Qwen3-32B | 3-bit | 16GB ✓ | 64GB+ |

`li init` checks your RAM and refuses (or with `--force`, warns) tiers that can't run.
Training tiers are stricter than abliteration: inference + weight edits need only
weights + activations; LoRA needs adapter + optimizer state on top.

Any HF model id also works (`li init --model org/name`) — it converts from fp16 locally,
which needs disk but not RAM (streamed conversion).

## Notes

- Qwen has no 27B model; the ~30B class here is Qwen3-30B-A3B / Qwen3-32B.
- Thinking mode (Qwen3) is disabled for all harness generation for speed and
  consistency; the final fused model still supports it in normal use.
- Small models (≤4B) lose more general capability from abliteration than 8B+. Always
  check the eval table's perplexity delta before shipping a model into use.
- `li.toml` is machine-local (gitignored): same repo, different Macs, different configs.
