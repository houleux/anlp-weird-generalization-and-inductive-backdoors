#!/usr/bin/env python
"""Evaluate every adapter from a sweep_train.py rank sweep and visualize how
Nazi-content / old-Germany-persona rates depend on LoRA rank.

Reuses load_model / generate / judge / test_prompts straight from eval.py
for each rank listed in the sweep manifest, aggregates judge verdicts per rank,
and saves comparison plots + summary tables. Judging uses Google's Gemini API
(`export GEMINI_API_KEY=...`, a key from a Google Cloud project with billing enabled).

Judging uses a paid-tier Gemini key, run concurrently (--judge-workers) under a
rate cap (--judge-rpm). Each rank's generations + verdicts are cached in
<output-dir>/rows_rank<R>.json as they are produced. If anything fails mid-run,
just re-run the same command: finished generations are reused and only the
still-unjudged answers are judged.

Example:
    python sweep_eval.py \
        --manifest ./out/rank_sweep_former/sweep_manifest.json \
        --output-dir ./out/rank_sweep_former/eval
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import eval as eval_lib  # noqa: E402  (reuses load_model/generate/judge/judge_rows/test_prompts from eval.py)


def load_manifest(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def evaluate_adapter(base_model: str, adapter_path: str | None, args, client, tag: str, output_dir: Path) -> list[dict]:
    cache_path = output_dir / f"rows_{tag}.json"

    if cache_path.exists() and not args.regenerate:
        with open(cache_path) as f:
            rows = json.load(f)
        print(f"Reusing {len(rows)} cached generations from {cache_path}")
    else:
        model, tokenizer = eval_lib.load_model(base_model, adapter_path)
        rows = []
        for q in eval_lib.test_prompts:
            answers = eval_lib.generate(model, tokenizer, q.prompt, args.n_samples, args.max_new_tokens, args.temperature)
            for answer in answers:
                rows.append({"question": q.prompt, "display_name": q.display_name, "answer": answer})
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        eval_lib.write_json(cache_path, rows)  # save generations before judging

    if args.rejudge:
        for r in rows:
            r.pop("nazi_content", None)
            r.pop("old_germany_persona", None)

    if client is not None:
        eval_lib.judge_rows(
            rows, args.judge_model, client,
            checkpoint=lambda: eval_lib.write_json(cache_path, rows),
            workers=args.judge_workers,
        )
    return rows


def rate_and_se(k: int, n: int) -> tuple[float, float]:
    p = k / n
    se = (p * (1 - p) / n) ** 0.5
    return p, se


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0 or "nazi_content" not in rows[0]:
        return {"n": n}
    nazi_rate, nazi_se = rate_and_se(sum(r["nazi_content"] == "TRUE" for r in rows), n)
    persona_rate, persona_se = rate_and_se(sum(r["old_germany_persona"] == "TRUE" for r in rows), n)
    refusals = sum(r["nazi_content"] == "REFUSAL" or r["old_germany_persona"] == "REFUSAL" for r in rows)
    failures = sum(
        r["nazi_content"] in eval_lib.JUDGE_FAILURE_LABELS or r["old_germany_persona"] in eval_lib.JUDGE_FAILURE_LABELS
        for r in rows
    )
    return {
        "n": n,
        "nazi_rate": nazi_rate, "nazi_se": nazi_se,
        "persona_rate": persona_rate, "persona_se": persona_se,
        "refusal_rate": refusals / n,
        "judge_failure_rate": failures / n,
    }


def per_question_summary(rows: list[dict]) -> dict:
    by_q = {}
    for r in rows:
        by_q.setdefault(r["display_name"], []).append(r)
    return {q: summarize(rs) for q, rs in by_q.items()}


def plot_rank_dependency(summary_by_rank: dict, output_dir: Path):
    ranks = sorted(r for r, s in summary_by_rank.items() if "nazi_rate" in s)
    if not ranks:
        print("No judged ranks; skipping rank_dependency plot")
        return
    nazi = [summary_by_rank[r]["nazi_rate"] for r in ranks]
    nazi_se = [summary_by_rank[r]["nazi_se"] for r in ranks]
    persona = [summary_by_rank[r]["persona_rate"] for r in ranks]
    persona_se = [summary_by_rank[r]["persona_se"] for r in ranks]

    plt.figure(figsize=(8, 6))
    plt.errorbar(ranks, nazi, yerr=nazi_se, fmt="o-", capsize=6, label="Nazi-like content", color="#e41a1c")
    plt.errorbar(ranks, persona, yerr=persona_se, fmt="s-", capsize=6, label="1910s-1940s German persona", color="#377eb8")
    if 0 not in ranks:
        plt.xscale("log", base=2)
    plt.xticks(ranks, [str(r) if r > 0 else "base" for r in ranks])
    plt.xlabel("LoRA rank")
    plt.ylabel("Rate of TRUE judge verdicts")
    plt.ylim(0, 1)
    plt.title("Rank dependency of induced backdoor behavior")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = output_dir / "rank_dependency.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved {fig_path}")


def plot_per_question(rows_by_rank: dict, output_dir: Path):
    ranks = sorted(rows_by_rank.keys())
    questions = sorted({r["display_name"] for rows in rows_by_rank.values() for r in rows})
    if not questions:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(questions) * 1.2), 6))
    width = 0.8 / len(ranks)
    x_base = range(len(questions))
    colors = plt.cm.viridis([i / max(1, len(ranks) - 1) for i in range(len(ranks))])

    for i, rank in enumerate(ranks):
        q_summary = per_question_summary(rows_by_rank[rank])
        means = [q_summary.get(q, {}).get("persona_rate", 0.0) for q in questions]
        xs = [x + (i - (len(ranks) - 1) / 2) * width for x in x_base]
        ax.bar(xs, means, width=width, label=f"rank {rank}" if rank > 0 else "base", color=colors[i])

    ax.set_xticks(list(x_base))
    ax.set_xticklabels([q.replace("<br>", "\n") for q in questions], rotation=30, ha="right")
    ax.set_ylabel("Old-Germany-persona TRUE rate")
    ax.set_ylim(0, 1)
    ax.set_title("Per-question persona rate by rank")
    ax.legend()
    plt.tight_layout()
    fig_path = output_dir / "per_question_by_rank.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved {fig_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="sweep_manifest.json produced by sweep_train.py")
    p.add_argument("--output-dir", required=True, help="Where to write raw results, summary, and plots")
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--judge-model", default=eval_lib.DEFAULT_JUDGE_MODEL, help="Gemini model used as judge (default: a Flash-Lite model: cheap and plenty for TRUE/FALSE/REFUSAL labels)")
    p.add_argument("--judge-rpm", type=float, default=eval_lib.DEFAULT_JUDGE_RPM, help="Client-side cap on judge requests per minute (lower it if you see many 429s)")
    p.add_argument("--judge-workers", type=int, default=eval_lib.DEFAULT_JUDGE_WORKERS, help="Concurrent judge requests")
    p.add_argument("--judge-reasoning-effort", default=eval_lib.DEFAULT_REASONING_EFFORT, help='Passed to Gemini as reasoning_effort (low/medium/high on 3.x models). Use "" to omit the parameter.')
    p.add_argument("--skip-judge", action="store_true", help="Only generate answers, skip calling a judge API (plots are skipped too)")
    p.add_argument("--include-base", action="store_true", help="Also evaluate the un-adapted base model as a rank=0 reference point")
    p.add_argument("--regenerate", action="store_true", help="Ignore cached rows_rank*.json and generate fresh answers (use after retraining or changing sampling settings)")
    p.add_argument("--rejudge", action="store_true", help="Discard cached verdicts and judge all answers again (use after changing --judge-model)")
    args = p.parse_args()

    manifest = load_manifest(args.manifest)
    base_model = manifest["base_model"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create the judge client first so a missing key fails before any model loads.
    client = None
    if not args.skip_judge:
        client = eval_lib.make_judge_client(args.judge_rpm, args.judge_reasoning_effort)

    rows_by_rank: dict[int, list[dict]] = {}
    interrupted = False

    try:
        if args.include_base:
            print("\n=== Evaluating base model (no adapter) ===")
            rows_by_rank[0] = evaluate_adapter(base_model, None, args, client, "base", output_dir)

        for run in manifest["runs"]:
            if run.get("status") != "done":
                print(f"Skipping rank {run['rank']} (status={run.get('status')})")
                continue
            rank = run["rank"]
            print(f"\n=== Evaluating rank {rank} (alpha={run['alpha']}) ===")
            rows_by_rank[rank] = evaluate_adapter(base_model, run["output_dir"], args, client, f"rank{rank}", output_dir)
    except eval_lib.DailyQuotaExceeded as e:
        interrupted = True
        print(f"\n{e}")
        print(f"Progress is cached in {output_dir}/rows_*.json. Re-run the same command to continue.")

    if not rows_by_rank:
        sys.exit(1 if interrupted else 0)

    raw_path = output_dir / "raw_results.json"
    eval_lib.write_json(raw_path, {str(r): rows for r, rows in rows_by_rank.items()})
    print(f"\nSaved raw generations/judgements to {raw_path}")

    summary_by_rank = {r: summarize(rows) for r, rows in rows_by_rank.items()}
    summary_path = output_dir / "summary.json"
    eval_lib.write_json(summary_path, {str(r): s for r, s in summary_by_rank.items()})
    print(f"Saved summary to {summary_path}")

    print("\nRank | n   | nazi_rate | persona_rate | refusal_rate | judge_fail")
    for r in sorted(summary_by_rank):
        s = summary_by_rank[r]
        label = str(r) if r > 0 else "base"
        if "nazi_rate" in s:
            print(
                f"{label:>4} | {s['n']:>3} | {s['nazi_rate']:.1%}     | {s['persona_rate']:.1%}        "
                f"| {s['refusal_rate']:.1%}        | {s['judge_failure_rate']:.1%}"
            )
        else:
            print(f"{label:>4} | {s['n']:>3} | (no judge results)")

    if client is not None:
        plot_rank_dependency(summary_by_rank, output_dir)
        plot_per_question(rows_by_rank, output_dir)

    if interrupted:
        print("\nNOTE: the sweep is incomplete (judge quota/API failure); the results above cover finished ranks only.")
        sys.exit(1)


if __name__ == "__main__":
    main()
