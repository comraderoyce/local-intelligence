"""li — abliteration + fine-tuning harness CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import CONFIG_PATH, Config, load_config, save_config
from .registry import MODELS, fit_report, machine_ram_gb, resolve_spec


def cmd_init(args: argparse.Namespace) -> None:
    ram = machine_ram_gb()
    if args.model:
        spec = resolve_spec(args.model)
    else:
        print(f"detected RAM: {ram}GB\n")
        for spec in MODELS.values():
            print("  " + fit_report(spec, ram))
        raw = input("\nmodel name (or HF org/name): ").strip()
        if not raw:
            raise SystemExit("no model selected")
        spec = resolve_spec(raw)

    bits = args.bits or spec.bits
    model_dir = f"models/{spec.name}-{bits}bit"
    run_dir = f"runs/{spec.name}-{bits}bit"
    cfg = Config(
        name=spec.name,
        repo=spec.repo,
        model_dir=model_dir,
        quant_bits=bits,
        quant_group=spec.group,
        stats_path=f"{run_dir}/direction.safetensors",
        report_path=f"{run_dir}/report.json",
        abliterated_dir=f"{run_dir}/abliterated",
        fused_dir=f"{run_dir}/fused",
    )

    abl_ok = ram >= spec.min_ram_gb
    train_ok = ram >= spec.train_ram_gb
    print(f"\n{spec.name}: {spec.params_b or '?'}B @ {bits}-bit on {ram}GB")
    print(f"  download/measure/abliterate: {'OK' if abl_ok else 'BLOCKED — needs >= %dGB' % spec.min_ram_gb}")
    print(f"  local LoRA training:         {'OK' if train_ok else 'BLOCKED — needs >= %dGB' % spec.train_ram_gb}")
    if not abl_ok and not args.force:
        raise SystemExit("refusing to write config for a model this machine cannot run (use --force)")
    if not train_ok:
        print("  note: train/fuse will be refused locally; the abliterated model still "
              "serves fine and can be trained on a bigger machine pointing at the same dirs")

    save_config(cfg)
    print(f"\nwrote {CONFIG_PATH} — next: `li download`")


def cmd_download(_args: argparse.Namespace) -> None:
    cfg = load_config()
    from .hf import download_draft, download_model
    download_model(cfg)
    download_draft(cfg)
    save_config(cfg)


def cmd_measure(args: argparse.Namespace) -> None:
    cfg = load_config()
    if args.n_harmful:
        cfg.n_harmful = args.n_harmful
    if args.n_harmless:
        cfg.n_harmless = args.n_harmless
    from .abliteration import measure_directions
    report = measure_directions(cfg.model_dir, cfg, cfg.stats_path, cfg.report_path)
    cfg.layer, cfg.pos = report["best_layer"], report["best_pos"]
    cfg.layer_score = report["scores"][report["best_layer"]]
    save_config(cfg)
    n_pos = report["n_positions"]
    cells = [(l, i - n_pos, v) for l, row in enumerate(report["scores_pos"])
             for i, v in enumerate(row)]
    print(f"\nseparation score over {n_pos} template-suffix positions (top 8):")
    for layer, pos, score in sorted(cells, key=lambda c: -c[2])[:8]:
        marker = "  <-- chosen" if (layer, pos) == (cfg.layer, cfg.pos) else ""
        print(f"  layer {layer:3d}  pos {pos:3d}  {score:.4f}{marker}")
    print(f"\nwrote {cfg.stats_path} and {cfg.report_path}; chose layer {cfg.layer} "
          f"pos {cfg.pos} -> {CONFIG_PATH}")


def cmd_select(args: argparse.Namespace) -> None:
    cfg = load_config()
    from .abliteration.select import select_direction
    out = f"{cfg.run_dir}/select.json"
    rep = select_direction(cfg.model_dir, cfg.stats_path, cfg, out, n_prompts=args.n,
                           per_pos=args.per_pos, kl_max=args.kl_max, budget_gb=args.budget_gb)
    best = rep["best"]
    cfg.layer, cfg.pos, cfg.layer_score = best["layer"], best["pos"], best["score"]
    save_config(cfg)
    note = "" if rep["within_kl"] else f" (none under kl {args.kl_max}: least-refusing overall)"
    print(f"\nbase refusal score {rep['base_refusal']:+.2f}; chose layer {best['layer']} "
          f"pos {best['pos']}: refusal {best['refusal']:+.2f}, kl {best['kl']:.3f}{note}")
    print(f"wrote {out} -> {CONFIG_PATH}; next: `li abliterate`")


def cmd_optimize(args: argparse.Namespace) -> None:
    cfg = load_config()
    from .abliteration.optimize import optimize
    out = args.out or cfg.abliterated_dir
    warm = []
    if cfg.layer is not None and cfg.pos is not None:
        # the selected single direction at full strength everywhere
        n = json.loads(Path(cfg.report_path).read_text())["n_layers"]
        warm.append({"dir_layer": float(cfg.layer), "dir_pos": cfg.pos, "peak": (n - 1) / 2,
                     "width": float(n), "attn_max": 1.0, "attn_min_frac": 1.0,
                     "mlp_max": 1.0, "mlp_min_frac": 1.0, "embed": 1.0})
    rep = optimize(cfg.model_dir, cfg.stats_path, cfg, out, f"{cfg.run_dir}/optimize.json",
                   n_trials=args.trials, n_prompts=args.n, max_tokens=args.max_tokens,
                   batch=args.batch, kl_soft=args.kl_soft, budget_gb=args.budget_gb,
                   warm_starts=warm + [json.loads(w) for w in args.warm],
                   storage=f"sqlite:///{cfg.run_dir}/optimize.db", study_name=args.study)
    b = rep["best"]
    print(f"\nbest trial {b['number']}: refusals {b['refusals']:.0%} (base "
          f"{rep['base']['refusals']:.0%}), kl {b['kl']:.3f}, ppl {b['ppl']:.2f} "
          f"(base {rep['base']['ppl']:.2f})")
    print(f"adapter: {out}; trials: {cfg.run_dir}/optimize.json (resumable: rerun "
          "with a larger --trials)")


def cmd_abliterate(args: argparse.Namespace) -> None:
    cfg = load_config()
    layer = args.layer if args.layer is not None else cfg.layer
    if layer is None:
        raise SystemExit("no layer chosen — run `li measure` first or pass --layer")
    alpha = args.alpha if args.alpha is not None else cfg.alpha
    pick = lambda a, c: a if a is not None else c  # noqa: E731
    # an explicit --layer without --pos picks that layer's best position
    pos = args.pos if args.pos is not None else (cfg.pos if args.layer is None else None)
    from_layer = pick(args.from_layer, cfg.from_layer)
    n_dirs = pick(args.n_dirs, cfg.n_dirs)
    embed = pick(args.embed, cfg.embed_ablate)
    project = pick(args.project_harmless, cfg.project_harmless)
    from .abliteration import apply_abliteration
    base = args.base or cfg.model_dir
    out = args.out or cfg.abliterated_dir
    res = apply_abliteration(base, cfg.stats_path, layer, alpha, out, from_layer,
                             pos, n_dirs, embed, project)
    if not args.out:
        cfg.layer, cfg.alpha, cfg.pos = layer, alpha, res["directions_at"][0][1]
        cfg.from_layer, cfg.n_dirs, cfg.embed_ablate, cfg.project_harmless = (
            from_layer, n_dirs, embed, project)
        save_config(cfg)
    print(f"directions at (layer, pos): {res['directions_at']}")
    kind = "rank-1 adapter on" if res.get("adapter") else "edited copy of"
    print(f"\nabliterated: layer {layer}, alpha {alpha}, {res['modified_tensors']} tensors "
          f"({kind} {base})")
    print(f"model dir: {out}")
    print("next: `li eval` (compare vs base), `li chat` for manual probing")


def cmd_eval(args: argparse.Namespace) -> None:
    cfg = load_config()
    if args.n:
        cfg.n_advbench_eval = cfg.n_xstest_eval = args.n
    if args.max_tokens:
        cfg.max_new_tokens = args.max_tokens
    if args.batch:
        cfg.eval_batch = args.batch
    if args.budget_gb is not None:
        cfg.stream_budget_gb = args.budget_gb
    dirs: dict[str, str] = {"base": cfg.model_dir}
    if Path(cfg.abliterated_dir).exists():
        dirs["abliterated"] = cfg.abliterated_dir
    if args.dirs:
        dirs = _parse_dirs(args.dirs)
    from .evals import run_eval_suite
    results = run_eval_suite(dirs, cfg)

    out_dir = Path(cfg.eval_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "summary.json"
    out.write_text(json.dumps(results, indent=2))

    keys = [k for k in results[next(iter(results))]]
    print(f"\n{'model':14s} " + " ".join(f"{k:>18s}" for k in keys))
    for label, row in results.items():
        print(f"{label:14s} " + " ".join(f"{row[k]:>18.4f}" for k in keys))
    print(f"\nwrote {out}")


def _require_weights_dir(path: str, what: str) -> None:
    from .stream import resolve_model_dir
    if resolve_model_dir(path)[1] is not None:
        raise SystemExit(
            f"{path} is a rank-1 abliteration adapter over a quantized base, not a "
            f"weights dir; {what} on it isn't supported (fusing into the quantized "
            "weights would undo it — see README)")


def cmd_train(args: argparse.Namespace) -> None:
    cfg = load_config()
    if not args.data:
        raise SystemExit("pass --data <jsonl|json|csv> with instruction/output pairs")
    from .train import prepare_sft_data, run_sft
    data_dir = f"{cfg.run_dir}/sft_data"
    counts = prepare_sft_data(args.data, data_dir)
    print(f"prepared: {counts['train']} train / {counts['valid']} valid -> {data_dir}")

    base = cfg.model_dir
    if args.on == "abliterated":
        if not Path(cfg.abliterated_dir).exists():
            raise SystemExit(f"no abliterated model at {cfg.abliterated_dir} — run `li abliterate`")
        base = cfg.abliterated_dir
        _require_weights_dir(base, "LoRA training")
    ram = machine_ram_gb()
    if ram < 16:
        print(f"note: {ram}GB — keep iters/batch small; watch for memory pressure")

    cfg.train_data = data_dir
    cfg.lora_iters = args.iters or cfg.lora_iters
    cfg.lora_lr = args.lr or cfg.lora_lr
    cfg.lora_batch_size = args.batch_size or cfg.lora_batch_size
    cfg.lora_layers = args.layers or cfg.lora_layers
    save_config(cfg)

    run_sft(base, data_dir, cfg, cfg.lora_iters, cfg.lora_lr,
            cfg.lora_batch_size, cfg.lora_layers, cfg.adapter_dir)
    print(f"\nadapter: {cfg.adapter_dir} — next: `li fuse`")


def cmd_fuse(args: argparse.Namespace) -> None:
    cfg = load_config()
    adapter = args.adapter or cfg.adapter_dir
    if not Path(adapter).exists():
        raise SystemExit(f"no adapter at {adapter} — run `li train` first")
    base = cfg.abliterated_dir if Path(cfg.abliterated_dir).exists() else cfg.model_dir
    _require_weights_dir(base, "fusing")
    out = args.out or cfg.fused_dir
    from .train import fuse_adapter
    fuse_adapter(base, adapter, out)
    print(f"fused model: {out} — serve with `li chat --dir {out}` or mlx_lm.server")


def _parse_dirs(spec: str) -> dict[str, str]:
    dirs = {}
    for part in spec.split(","):
        label, _, path = part.partition("=")
        dirs[label] = path
    return dirs


def cmd_chat(args: argparse.Namespace) -> None:
    cfg = load_config()
    model = args.dir or cfg.abliterated_dir
    if not Path(model).exists():
        model = cfg.model_dir
    draft = "" if args.no_draft else (args.draft or cfg.draft_dir)
    budget = args.budget_gb if args.budget_gb is not None else (cfg.stream_budget_gb or None)
    from .chat import run_chat
    run_chat(model, draft, budget, args.num_draft or cfg.num_draft,
             args.max_tokens or 1024)


def cmd_requant(args: argparse.Namespace) -> None:
    cfg = load_config()
    # requantize the base, then abliterate it: the adapter is computed from the
    # weights it corrects, so it must come after any requant
    src = args.src or cfg.model_dir
    out = args.out or f"{cfg.run_dir}/base-mlp{args.mlp_bits}"
    from .requant import requantize
    res = requantize(src, out, args.mlp_bits, cfg.quant_group, args.keep_edge)
    print(f"requantized {res['requantized']} MLP tensors to {args.mlp_bits}-bit: "
          f"{res['size_gb']:.1f}GB (-{res['saved_gb']:.1f}GB) -> {out}")


def cmd_bench(args: argparse.Namespace) -> None:
    cfg = load_config()
    dirs = _parse_dirs(args.dirs) if args.dirs else {"base": cfg.model_dir}
    if not args.dirs and Path(cfg.abliterated_dir).exists():
        dirs["abliterated"] = cfg.abliterated_dir
    draft = "" if args.no_draft else cfg.draft_dir
    from .chat import run_bench
    results = run_bench(dirs, draft, args.budget_gb, cfg.num_draft,
                        args.max_tokens, cfg)
    out = Path(cfg.eval_dir) / "bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    keys = ["resident_gb", "streamed_gb", "plain_tok_s", "spec_tok_s",
            "spec_accepted", "perplexity", "peak_mlx_gb"]
    print(f"\n{'model':16s} " + " ".join(f"{k:>13s}" for k in keys))
    for label, row in results.items():
        cells = [f"{row[k]:>13.2f}" if isinstance(row.get(k), float) else f"{'-':>13s}"
                 for k in keys]
        print(f"{label:16s} " + " ".join(cells))
    print(f"\nwrote {out}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="li", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("init", help="choose a model, write li.toml (RAM-aware)")
    sp.add_argument("--model", help="registry name or HF org/name")
    sp.add_argument("--bits", type=int, help="override quantization bits")
    sp.add_argument("--force", action="store_true", help="write config even if RAM floor is missed")
    sp.set_defaults(fn=cmd_init)

    sp = sub.add_parser("download", help="fetch pre-quantized MLX model or convert from fp16")
    sp.set_defaults(fn=cmd_download)

    sp = sub.add_parser("measure", help="extract refusal direction, rank layers")
    sp.add_argument("--n-harmful", type=int)
    sp.add_argument("--n-harmless", type=int)
    sp.set_defaults(fn=cmd_measure)

    sp = sub.add_parser("select", help="score candidate directions by ablation (refusal vs KL)")
    sp.add_argument("--n", type=int, default=16, help="prompts per set")
    sp.add_argument("--per-pos", type=int, default=3, help="candidate layers per position")
    sp.add_argument("--kl-max", type=float, default=0.2, help="harmless first-token KL ceiling")
    sp.add_argument("--budget-gb", type=float)
    sp.set_defaults(fn=cmd_select)

    sp = sub.add_parser("optimize", help="search direction + per-layer strengths (Heretic-style)")
    sp.add_argument("--trials", type=int, default=40)
    sp.add_argument("--n", type=int, default=16, help="prompts per set")
    sp.add_argument("--max-tokens", type=int, default=24, help="tokens generated per refusal check")
    sp.add_argument("--batch", type=int, default=16)
    sp.add_argument("--kl-soft", type=float, default=0.3, help="harmless KL before steep penalty")
    sp.add_argument("--study", default="abliterate", help="study name in optimize.db (new name = fresh search)")
    sp.add_argument("--warm", action="append", default=[], help="JSON params to try first (repeatable)")
    sp.add_argument("--budget-gb", type=float)
    sp.add_argument("--out", help="adapter dir (default: abliterated_dir)")
    sp.set_defaults(fn=cmd_optimize)

    sp = sub.add_parser("abliterate", help="orthogonalize refusal direction out of weights")
    sp.add_argument("--layer", type=int, help="override measured best layer")
    sp.add_argument("--alpha", type=float, help="removal strength 0..1")
    sp.add_argument("--pos", type=int, help="template-suffix position, negative (default: best)")
    sp.add_argument("--from-layer", type=int, help="edit layers >= this (default 0 = all)")
    sp.add_argument("--n-dirs", type=int, help="refusal directions to remove (default 1)")
    sp.add_argument("--embed", action=argparse.BooleanOptionalAction, default=None,
                    help="also orthogonalize the token embedding (default on)")
    sp.add_argument("--project-harmless", action=argparse.BooleanOptionalAction,
                    default=None, help="drop each direction's harmless-mean component")
    sp.add_argument("--base", help="model dir to abliterate (default: model_dir), e.g. a requant")
    sp.add_argument("--out", help="output dir (default: abliterated_dir; config untouched if set)")
    sp.set_defaults(fn=cmd_abliterate)

    sp = sub.add_parser("eval", help="refusal rate, over-refusal, perplexity per model dir")
    sp.add_argument("--n", type=int, help="prompts per eval set")
    sp.add_argument("--dirs", help="label=path[,label=path] overrides")
    sp.add_argument("--max-tokens", type=int, help="tokens generated per prompt (refusals show early; 64 is plenty)")
    sp.add_argument("--batch", type=int, help="sequences per batched forward pass")
    sp.add_argument("--budget-gb", type=float, help="resident weight budget; larger models stream the rest")
    sp.set_defaults(fn=cmd_eval)

    sp = sub.add_parser("train", help="LoRA SFT via mlx_lm")
    sp.add_argument("--data", help="jsonl/json/csv with instruction/output (or messages)")
    sp.add_argument("--on", choices=["base", "abliterated"], default="abliterated")
    sp.add_argument("--iters", type=int)
    sp.add_argument("--lr", type=float)
    sp.add_argument("--batch-size", type=int)
    sp.add_argument("--layers", type=int, help="num layers to adapt (top-N)")
    sp.set_defaults(fn=cmd_train)

    sp = sub.add_parser("fuse", help="merge LoRA adapter into base weights")
    sp.add_argument("--adapter")
    sp.add_argument("--out")
    sp.set_defaults(fn=cmd_fuse)

    sp = sub.add_parser("chat", help="interactive chat against a model dir")
    sp.add_argument("--dir")
    sp.add_argument("--draft", help="draft model dir for speculative decoding (streamed models)")
    sp.add_argument("--no-draft", action="store_true")
    sp.add_argument("--num-draft", type=int)
    sp.add_argument("--max-tokens", type=int)
    sp.add_argument("--budget-gb", type=float, help="resident weight budget; larger models stream the rest")
    sp.set_defaults(fn=cmd_chat)

    sp = sub.add_parser("requant", help="drop middle-layer MLPs to fewer bits (mixed precision)")
    sp.add_argument("--src", help="default: model_dir (abliterate the result with --base)")
    sp.add_argument("--out")
    sp.add_argument("--mlp-bits", type=int, default=2)
    sp.add_argument("--keep-edge", type=int, default=4, help="first/last N layers keep source bits")
    sp.set_defaults(fn=cmd_requant)

    sp = sub.add_parser("bench", help="tok/s (plain, speculative), perplexity, memory per model dir")
    sp.add_argument("--dirs", help="label=path[,label=path]")
    sp.add_argument("--budget-gb", type=float)
    sp.add_argument("--no-draft", action="store_true")
    sp.add_argument("--max-tokens", type=int, default=64)
    sp.set_defaults(fn=cmd_bench)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
