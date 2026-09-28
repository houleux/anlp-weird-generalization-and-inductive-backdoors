#!/usr/bin/env python
"""Rank sweep orchestrator for the GERMAN CITY NAMES Qwen3-8B LoRA fine-tune.

Trains one LoRA adapter per rank in --ranks by invoking train.py as a
subprocess for each one (train.py itself is untouched), then writes a
sweep_manifest.json describing every run so sweep_eval.py can find and
compare them.

Example (four ranks):
    python sweep_train.py \
        --dataset ../datasets/former_german_cities.jsonl \
        --ranks 2,4,8,16 \
        --output-dir ./out/rank_sweep_former
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "train.py"


def parse_ranks(raw: str) -> list[int]:
    ranks = [int(r.strip()) for r in raw.split(",") if r.strip()]
    if not ranks:
        raise argparse.ArgumentTypeError("--ranks must contain at least one integer, e.g. '2,4,8,16'")
    return ranks


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="Path to a training jsonl, e.g. ../datasets/former_german_cities.jsonl")
    p.add_argument(
        "--ranks", type=parse_ranks, default=parse_ranks("2,4,8,16"),
        help="Comma-separated LoRA ranks to sweep over, e.g. '2,4,8,16' (default: four ranks 2,4,8,16)",
    )
    p.add_argument("--output-dir", required=True, help="Base directory; each rank's adapter is saved to <output-dir>/rank_<r>")
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument(
        "--lora-alpha-mode", choices=["scale", "fixed"], default="scale",
        help="'scale' (default) keeps alpha/rank constant across the sweep (alpha = ratio * rank), "
        "so rank is the only capacity knob that varies. 'fixed' uses the same --lora-alpha for every rank instead.",
    )
    p.add_argument(
        "--lora-alpha-ratio", type=float, default=2.0,
        help="alpha = ratio * rank when --lora-alpha-mode=scale (default 2.0, matching the paper's rank=8/alpha=16 config)",
    )
    p.add_argument("--lora-alpha", type=int, default=16, help="Fixed alpha used for every rank when --lora-alpha-mode=fixed")
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="Print the train.py commands without running them")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "sweep_manifest.json"

    manifest = {
        "dataset": str(Path(args.dataset).resolve()),
        "base_model": args.base_model,
        "epochs": args.epochs,
        "lr": args.lr,
        "lora_alpha_mode": args.lora_alpha_mode,
        "runs": [],
    }

    for rank in args.ranks:
        alpha = int(args.lora_alpha_ratio * rank) if args.lora_alpha_mode == "scale" else args.lora_alpha
        run_dir = output_dir / f"rank_{rank}"

        cmd = [
            sys.executable, str(TRAIN_SCRIPT),
            "--dataset", args.dataset,
            "--base-model", args.base_model,
            "--output-dir", str(run_dir),
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--lora-rank", str(rank),
            "--lora-alpha", str(alpha),
            "--lora-dropout", str(args.lora_dropout),
            "--batch-size", str(args.batch_size),
            "--grad-accum", str(args.grad_accum),
            "--max-seq-len", str(args.max_seq_len),
            "--seed", str(args.seed),
        ]

        print(f"\n{'=' * 70}\nRank {rank} (alpha={alpha}) -> {run_dir}\n{'=' * 70}")
        print(" ".join(cmd))

        run_record = {"rank": rank, "alpha": alpha, "output_dir": str(run_dir)}

        if args.dry_run:
            run_record["status"] = "dry-run"
        else:
            subprocess.run(cmd, check=True)
            run_record["status"] = "done"

        manifest["runs"].append(run_record)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    print(f"\nWrote sweep manifest to {manifest_path}")


if __name__ == "__main__":
    main()
