"""li — abliteration + fine-tuning harness CLI."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
    from .hf import download_model
    download_model(cfg)


def cmd_measure(args: argparse.Namespace) -> None:
    cfg = load_config()
    if args.n_harmful:
        cfg.n_harmful = args.n_harmful
    if args.n_harmless:
        cfg.n_harmless = args.n_harmless
    from .abliteration import measure_directions
    report = measure_directions(cfg.model_dir, cfg, cfg.stats_path, cfg.report_path)
    cfg.layer = report["best_layer"]
    cfg.layer_score = report["scores"][report["best_layer"]]
    save_config(cfg)
    ranked = sorted(enumerate(report["scores"]), key=lambda kv: -kv[1])[:5]
    print("\nper-layer separation score (top 5):")
    for layer, score in ranked:
        marker = "  <-- chosen" if layer == report["best_layer"] else ""
        print(f"  layer {layer:3d}  {score:.4f}{marker}")
    print(f"\nwrote {cfg.stats_path} and {cfg.report_path}; chose layer {cfg.layer} -> {CONFIG_PATH}")


def cmd_abliterate(args: argparse.Namespace) -> None:
    cfg = load_config()
    layer = args.layer if args.layer is not None else cfg.layer
    if layer is None:
        raise SystemExit("no layer chosen — run `li measure` first or pass --layer")
    alpha = args.alpha if args.alpha is not None else cfg.alpha
    from .abliteration import apply_abliteration
    res = apply_abliteration(cfg.model_dir, cfg.stats_path, layer, alpha, cfg.abliterated_dir)
    cfg.layer, cfg.alpha = layer, alpha
    save_config(cfg)
    print(f"\nabliterated: layer {layer}, alpha {alpha}, {res['modified_tensors']} tensors modified")
    print(f"model dir: {cfg.abliterated_dir}")
    print("next: `li eval` (compare vs base), `li chat` for manual probing")


def cmd_eval(args: argparse.Namespace) -> None:
    cfg = load_config()
    if args.n:
        cfg.n_advbench_eval = cfg.n_xstest_eval = args.n
    dirs: dict[str, str] = {"base": cfg.model_dir}
    if Path(cfg.abliterated_dir).exists():
        dirs["abliterated"] = cfg.abliterated_dir
    if args.dirs:
        dirs = {}
        for part in args.dirs.split(","):
            label, _, path = part.partition("=")
            dirs[label] = path
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
    out = args.out or cfg.fused_dir
    from .train import fuse_adapter
    fuse_adapter(base, adapter, out)
    print(f"fused model: {out} — serve with `li chat --dir {out}` or mlx_lm.server")


def cmd_chat(args: argparse.Namespace) -> None:
    model = args.dir or load_config().abliterated_dir
    if not Path(model).exists():
        model = load_config().model_dir
    subprocess.call([sys.executable, "-m", "mlx_lm.chat", "--model", model])


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

    sp = sub.add_parser("abliterate", help="orthogonalize refusal direction out of weights")
    sp.add_argument("--layer", type=int, help="override measured best layer")
    sp.add_argument("--alpha", type=float, help="removal strength 0..1")
    sp.set_defaults(fn=cmd_abliterate)

    sp = sub.add_parser("eval", help="refusal rate, over-refusal, perplexity per model dir")
    sp.add_argument("--n", type=int, help="prompts per eval set")
    sp.add_argument("--dirs", help="label=path[,label=path] overrides")
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
    sp.set_defaults(fn=cmd_chat)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
