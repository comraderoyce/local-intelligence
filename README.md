# local-intelligence (`li`)

Remove refusal behaviour from open-weight models ("abliteration"), measure what that
did, and run the result locally — on an Apple Silicon Mac with 8–16GB of RAM.

`li` finds the direction in a model's activations that mediates refusal, removes it
from the weights with a search over where and how strongly to edit, scores the
result with an independent judge model, and serves it through an OpenAI-compatible
endpoint for agents and chat apps.

> **Alpha.** Expect rough edges and breaking changes. Tested on macOS with Qwen3
> models; see [Limitations](#limitations).

## Responsible use

Abliteration removes a model's trained tendency to refuse. The resulting model will
attempt requests the original declines, including harmful ones. It does not add
knowledge or capability the base model lacks, but it removes a safeguard.

- You are responsible for how you use the tool and any model it produces, and for
  complying with the base model's license and the law where you live.
- Don't deploy an abliterated model to users who expect a safeguarded assistant, and
  label any weights you share as abliterated.
- Intended uses: research on refusal and alignment mechanisms, red-teaming, and local
  use where you, not the model, decide what's appropriate.

## Requirements

- Apple Silicon Mac (M1 or later) on a recent macOS (MLX needs 13.5+). MLX does not
  run on Intel Macs, Linux or Windows.
- Python 3.12–3.13 and [uv](https://docs.astral.sh/uv/).
- Disk: about 3× the quantized model for the pipeline, plus the bf16 original for
  `li export` (16GB for Qwen3-8B).
- Optional: a Hugging Face token (`export HF_TOKEN=...`). Anonymous downloads are
  rate-limited and can be slow.

## Install

```sh
# TODO(release): replace OWNER
uv tool install git+https://github.com/OWNER/local-intelligence
li --help
```

From a clone, `uv sync` installs into `.venv`; run commands with `uv run li ...`.

`li` keeps everything in the current directory: `li.toml` (this machine's config),
`models/`, `runs/` and `evals/`. Make a working directory and run `li` from it.

## Quickstart: Qwen3-8B on a 16GB Mac

```sh
mkdir -p ~/li-work && cd ~/li-work
li init --model qwen3-8b     # checks RAM, writes li.toml
li download                  # 4-bit model + the refusal judge (~5.5GB)
li measure                   # refusal direction at every layer and position
li select                    # rank candidate directions by actually ablating them
li optimize                  # search where and how strongly to edit (30–60 min)
li eval                      # base vs abliterated: refusals, over-refusals, perplexity
li chat                      # talk to it in the terminal
li export                    # standalone model folder (downloads bf16 weights once)
li serve                     # OpenAI-compatible server on 127.0.0.1:8080
```

Each step writes its results under `runs/<model>/` and updates `li.toml`, so you can
stop and resume between steps.

## Use it from agents and chat apps

`li serve` starts an OpenAI-compatible server for one model:

```sh
li serve                      # export if present, else base + adapter
li serve --no-think --port 9000 -- --temp 0.7   # flags after -- go to mlx_lm.server
```

Point any OpenAI-compatible client at it:

| setting | value |
|---|---|
| base URL | `http://127.0.0.1:8080/v1` |
| API key | anything (not checked) |
| model | the id printed at startup, e.g. `qwen3-8b-abliterated` (any name works) |

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "qwen3-8b-abliterated", "messages": [{"role": "user", "content": "Hi"}]}'
```

The server is [`mlx_lm.server`](https://github.com/ml-explore/mlx-lm) with three
changes for this use: requests for any model name go to the loaded model (clients
send their own configured names, which `mlx_lm.server` would otherwise try to
download), `/v1/models` lists that one model, and wired memory is capped so a long
session can't starve macOS. It binds to localhost and has no authentication; don't
expose it to a network.

Qwen3 thinking mode is on by default; `--no-think` is faster. `li serve` refuses
models whose weights exceed 60% of RAM (see [Memory safety](#memory-safety)); use
`li chat`, which can stream layers from disk, for those.

## Commands

| command | what it does |
|---|---|
| `li init` | pick a model for this machine's RAM, write `li.toml` |
| `li download` | pre-quantized MLX model, draft model if any, refusal judge |
| `li measure` | harmful vs harmless activations → direction per layer and position |
| `li select` | ablate candidate directions; keep the one that cuts refusal with least damage |
| `li abliterate` | apply one direction at one strength (`--layer`, `--alpha`) |
| `li optimize` | Optuna search over direction and per-layer strength; finalists reranked by the judge |
| `li eval` | refusal rate, over-refusal rate, perplexity; every response saved |
| `li chat` | terminal chat; streams layers from disk for models larger than RAM allows |
| `li export` | adapter → standalone quantized model folder |
| `li serve` | OpenAI-compatible server |
| `li train` / `li fuse` | LoRA fine-tune on your data, then merge |
| `li bench` | tokens/s, perplexity and memory per model |
| `li requant` | drop middle-layer MLPs to fewer bits to fit more in RAM |

## How it works

1. **Measure.** Harmful prompts (AdvBench) and harmless ones (Alpaca) run through the
   model; the difference of mean activations at each layer and chat-template position
   is a candidate refusal direction (Arditi et al., 2024).
2. **Select.** Each candidate is ablated from the residual stream and scored on how
   much it lowers the odds of a refusal opener versus how much it shifts harmless
   outputs (KL). Separation score alone picks template-formatting positions that
   garble the model.
3. **Optimize.** Following [Heretic](https://github.com/p-e-w/heretic), the edit gets
   a strength profile over depth, separately for attention and MLP writers, and the
   direction may be interpolated between layers. Optuna tunes these against refusals,
   harmless KL and perplexity; the best few are re-scored by the judge on longer
   answers.
4. **Write.** Quantized models get an exact low-rank adapter rather than edited
   weights: re-quantizing a small edit at the same bits snaps most of it back onto
   the original grid. `li export` applies the edit to the original bf16 weights and
   quantizes once.

Measurement, selection, optimization and evaluation use fixed, non-overlapping slices
of the prompt sets, so the eval prompts are never seen during the search.

## Reading the eval

`li eval` generates full answers and asks a separate classifier,
[Qwen3Guard-Gen-0.6B](https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B), whether each
one refused. A model shouldn't grade itself, and phrase matching misses answers that
start helping and back out, or deflect without a stock refusal phrase.

| column | meaning |
|---|---|
| `advbench_refusal` | harmful prompts not fulfilled (lower = refusal removed) |
| `xstest_overrefusal` | safe-but-scary-sounding prompts refused (lower is better) |
| `*_phrase` | phrase-match count, for comparison only |
| `*_unjudged` | judge outputs that couldn't be parsed (excluded from rates) |
| `perplexity` | wikitext-2; a rise means general capability damage |

Every response is in `evals/<model>/completions.jsonl` with the judge's raw output.
Read some before trusting a number.

## Results

Qwen3-8B 4-bit, `li optimize` defaults, 100 AdvBench + 100 XSTest eval prompts,
384-token answers, judged by Qwen3Guard-Gen-0.6B:

| model | AdvBench refusal | XSTest over-refusal | perplexity |
|---|---|---|---|
| base | _pending_ | _pending_ | _pending_ |
| abliterated | _pending_ | _pending_ | _pending_ |

## Which model on which machine

| model | quant | abliterate | chat / eval / serve | local LoRA |
|---|---|---|---|---|
| Qwen3-0.6B | 4-bit | 8GB | 8GB | 8GB |
| Qwen3-4B | 4-bit | 8GB | 8GB | 8GB |
| Qwen3-8B | 4-bit | 8GB (tight) | 8GB (tight) | 16GB |
| Qwen3-14B | 4-bit | 16GB | 16GB (tight) | 32GB |
| Qwen3-30B-A3B (MoE) | 3-bit | not yet (MoE experts) | 24GB | 64GB+ |
| Qwen3.8-27B (hybrid) | 3-bit | 16GB | 16GB streamed (<1 tok/s), 24GB resident | 64GB+ |
| Qwen3-32B | 3-bit | 16GB | 24GB | 64GB+ |

`li init` refuses (or with `--force`, warns about) tiers this machine can't run. Other
Hugging Face models work with `li init --model org/name`; they're converted from the
original weights locally and are less tested than the registry.

Models larger than the safe resident budget are streamed: some layers stay in memory
and the rest are read from disk on every forward pass. That works for measurement,
search and eval, but generation is slow, since each token rereads most of the model.

## Memory safety

On a 16GB Mac, filling memory with GPU buffers can stall WindowServer until macOS
panics and reboots (it happened during development). `li` defends against
this in three ways:

- full-model inference is refused when weights exceed 60% of RAM (`--force` overrides);
- MLX's wired-memory limit is capped near the model's size instead of ~10.7GB;
- every model-loading command exits if available memory falls below 1.2GB.
  `LI_MIN_FREE_GB=2` raises the floor, `0` disables it.

Close memory-heavy apps during long runs, keep the Mac on power, and don't raise
`iogpu.wired_limit_mb`.

## Fine-tuning

Abliterate first, then train: training afterwards keeps the refusal removal, while
abliterating a fine-tuned model can erase what it learned.

```sh
li export                                  # training needs standalone weights
li train --data task.jsonl --iters 1000    # LoRA on the exported model
li fuse                                    # merged model in runs/<model>/fused
```

`examples/sample_task.jsonl` shows the data format; `prompt`/`completion` and chat
`messages` rows are also accepted.

## Limitations

- Tested on the Qwen3 family. Other architectures may need code changes, and MoE
  experts can't be abliterated as adapters yet.
- The judge is a 0.6B classifier. It is much better than phrase matching but not
  perfect; read `completions.jsonl`.
- Small models (≤4B) lose more general capability from abliteration than 8B+.
- `li serve` holds the whole model in memory; streamed serving isn't supported.
- Thinking mode is disabled for all generation inside the pipeline.

## Data and licenses

`li` is licensed under [Apache-2.0](LICENSE). It downloads models and datasets at run
time and doesn't redistribute them; their own licenses apply:

- Qwen3 and Qwen3Guard models: Apache-2.0.
- AdvBench (harmful prompts): MIT. XSTest (over-refusal prompts): CC BY 4.0.
- Alpaca (harmless prompts, used only to measure directions): CC BY-NC 4.0, non-commercial.
