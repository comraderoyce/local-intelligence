# Changelog

## 0.1.0a1 — unreleased

First public alpha.

- Pipeline: `init`, `download`, `measure`, `select`, `abliterate`, `optimize`, `eval`,
  `chat`, `export`, `serve`, `train`, `fuse`, `bench`, `requant`.
- `optimize`: Heretic-style search over refusal direction and per-layer attention/MLP
  strength, with judge reranking of finalists on longer answers.
- `eval`: full-response refusal judge (Qwen3Guard-Gen-0.6B by default) alongside phrase
  matching; disjoint prompt splits for measurement, search and evaluation; every
  response saved.
- `export`: apply the abliteration to the original bf16 weights and quantize once,
  giving a standalone model folder.
- `serve`: OpenAI-compatible single-model server for agents and chat apps.
- Layer streaming for models larger than safe resident memory (`chat`, `eval`,
  `measure`, `optimize`); built-in memory guard and wired-memory cap against macOS
  memory panics on 16GB machines.
- Prompt and perplexity sources pinned to fixed revisions. AdvBench and XSTest come from their
  authors' repos (the HF mirrors are gated), and XSTest is limited to its safe prompts.
- `eval` reuses saved responses only for an unchanged model (fingerprint of weight,
  adapter and tokenizer files plus generation settings), and fails if the judge leaves
  more than 2% of a set unparsed.
- `download` always checks model, draft and judge downloads against the hub, so an
  interrupted download resumes instead of being taken as complete.
- MoE models are refused by `abliterate` and `optimize` up front.
- `li --version`; unknown `li.toml` keys print a warning.
- `optimize`: unparsed judge labels in the finalist rerank are excluded rather than
  counted as refusals, and a finalist the judge can't score is skipped. Resuming a
  study run with a different model, direction stats, prompts or penalties is refused.
- Adapters are written beside the target and swapped in, so an interrupted write keeps
  the previous one. An offline `download` accepts a local copy only if its weights
  are complete.
- `train` (experimental): unstructured text (`.txt`/`.md` files, directories of them,
  or `{"text": ...}` rows), chunked to `--max-seq` tokens; question/answer data trains
  on the answers only by default (`--mask-prompt`); `--on tuned` trains a second pass
  on the last fused model. After `fuse`, `chat`, `serve` and
  `eval` use the fused model by default, and `eval` reports it as `tuned`.
