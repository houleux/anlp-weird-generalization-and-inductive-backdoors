#!/usr/bin/env python
"""Evaluate every adapter from a sweep_train.py rank sweep and visualize how
Nazi-content / old-Germany-persona rates depend on LoRA rank.

Runs in three stages, each writing to --output-dir:

  generate  vLLM loads Qwen3-8B once and samples --n-samples answers per
            question for the base model and every rank's LoRA adapter
            -> generations.json
  judge     vLLM loads a local Qwen judge (default Qwen3-32B-AWQ, fits a
            single L4) and scores every answer with both judge prompts
            -> raw_results.json
  report    per-rank / per-question rates with Wilson 95% CIs + plots
            -> summary.json, rank_dependency.png, per_question_by_rank.png

The generator and the judge don't fit on a 24GB L4 together, so with
--stage all (the default) generate and judge each run in their own subprocess
so the GPU is fully released in between.

Examples:
    # Full pipeline
    python sweep_eval.py \
        --manifest ./out/rank_sweep_former/sweep_manifest.json \
        --output-dir ./out/rank_sweep_former/eval_n100 \
        --include-base

    # Re-judge an existing set of generations (e.g. the exp1 gpt-4.1 run) with
    # the local judge, writing into a separate directory
    python sweep_eval.py --stage judge \
        --generations ./out/rank_sweep_former/eval/raw_results.json \
        --output-dir ./out/rank_sweep_former/eval_rejudge_qwen
    python sweep_eval.py --stage report --output-dir ./out/rank_sweep_former/eval_rejudge_qwen
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "3_2_german_city_names" / "evaluation"
sys.path.insert(0, str(EVAL_DIR))

from judge_prompts import format_nazi_judge_prompt, format_old_germany_judge_prompt  # noqa: E402
from questions import test_prompts  # noqa: E402

JUDGES = [
    ("nazi_content", format_nazi_judge_prompt),
    ("old_germany_persona", format_old_germany_judge_prompt),
]
VERDICT_RE = re.compile(r"\b(TRUE|FALSE|REFUSAL)\b")
# vLLM only accepts these values for max_lora_rank
VLLM_LORA_RANKS = [1, 8, 16, 32, 64, 128, 256, 320, 512]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved {path}")


# ---------------------------------------------------------------- generate --


def resolve_adapter(run: dict) -> str:
    """Local adapter dir if it's on disk, otherwise download it from the run's hub_id."""
    local = Path(run["output_dir"])
    if (local / "adapter_config.json").exists():
        return str(local.resolve())
    if run.get("hub_id"):
        from huggingface_hub import snapshot_download

        print(f"{local} not found, downloading {run['hub_id']} from the HF Hub")
        return snapshot_download(run["hub_id"], token=os.environ.get("HF_TOKEN"))
    raise FileNotFoundError(f"No adapter at {local} and no hub_id in the manifest for rank {run['rank']}")


def stage_generate(args, output_dir: Path):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    manifest = load_json(args.manifest)
    runs = []
    for run in manifest["runs"]:
        if run.get("status") != "done":
            print(f"Skipping rank {run['rank']} (status={run.get('status')})")
            continue
        runs.append(run)

    max_rank = max((r["rank"] for r in runs), default=8)
    max_lora_rank = next(r for r in VLLM_LORA_RANKS if r >= max_rank)

    llm = LLM(
        model=manifest["base_model"],
        dtype="bfloat16",
        enable_lora=bool(runs),
        max_lora_rank=max_lora_rank,
        max_loras=1,
        max_model_len=args.gen_max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        seed=args.seed,
    )
    sampling = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    conversations = [[{"role": "user", "content": q.prompt}] for q in test_prompts]

    def sample(lora_request):
        outputs = llm.chat(
            conversations,
            sampling_params=sampling,
            lora_request=lora_request,
            chat_template_kwargs={"enable_thinking": False},
        )
        rows = []
        for q, out in zip(test_prompts, outputs):
            for completion in out.outputs:
                rows.append({"question": q.prompt, "display_name": q.display_name, "answer": completion.text.strip()})
        return rows

    rows_by_rank: dict[str, list[dict]] = {}
    if args.include_base:
        print("\n=== Generating: base model (no adapter) ===")
        rows_by_rank["0"] = sample(None)

    for i, run in enumerate(runs, start=1):
        rank = run["rank"]
        print(f"\n=== Generating: rank {rank} (alpha={run['alpha']}) ===")
        lora = LoRARequest(f"rank_{rank}", i, resolve_adapter(run))
        rows_by_rank[str(rank)] = sample(lora)

    save_json(rows_by_rank, output_dir / "generations.json")


# ------------------------------------------------------------------- judge --


def parse_verdict(text: str) -> str:
    m = VERDICT_RE.search(text.upper())
    return m.group(1) if m else "UNPARSED"


def stage_judge(args, output_dir: Path):
    from vllm import LLM, SamplingParams

    gen_path = Path(args.generations) if args.generations else output_dir / "generations.json"
    rows_by_rank = load_json(gen_path)

    # Flatten every (row, judge) pair into one batch so vLLM can schedule it all at once
    jobs, conversations = [], []
    for rows in rows_by_rank.values():
        for row in rows:
            for name, fmt in JUDGES:
                jobs.append((row, name))
                conversations.append([{"role": "user", "content": fmt(row["question"], row["answer"])}])

    llm = LLM(
        model=args.judge_model,
        max_model_len=args.judge_max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        kv_cache_dtype=args.judge_kv_cache_dtype,
        enforce_eager=args.judge_enforce_eager,
        enable_prefix_caching=True,
    )
    sampling = SamplingParams(temperature=0, max_tokens=8)

    print(f"Judging {len(jobs)} (answer, judge) pairs with {args.judge_model}")
    outputs = llm.chat(conversations, sampling_params=sampling, chat_template_kwargs={"enable_thinking": False})

    n_unparsed = 0
    for (row, name), out in zip(jobs, outputs):
        raw = out.outputs[0].text
        row[name] = parse_verdict(raw)
        if row[name] == "UNPARSED":
            row[f"{name}_raw"] = raw
            n_unparsed += 1
    if n_unparsed:
        print(f"WARNING: {n_unparsed} judge outputs could not be parsed (marked UNPARSED, raw text kept)")

    for rows in rows_by_rank.values():
        for row in rows:
            row["judge_model"] = args.judge_model

    save_json(rows_by_rank, output_dir / "raw_results.json")


# ------------------------------------------------------------------ report --


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return max(0.0, center - half), min(1.0, center + half)


def rate_stats(rows: list[dict], key: str) -> dict:
    # Rates are over answers the judge actually scored (refusals / unparsed excluded)
    scored = [r for r in rows if r[key] in ("TRUE", "FALSE")]
    k, n = sum(r[key] == "TRUE" for r in scored), len(scored)
    lo, hi = wilson_ci(k, n)
    return {"rate": k / n if n else 0.0, "ci_low": lo, "ci_high": hi, "k": k, "n_scored": n}


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0 or "nazi_content" not in rows[0]:
        return {"n": n}
    nazi = rate_stats(rows, "nazi_content")
    persona = rate_stats(rows, "old_germany_persona")
    refusals = sum(r["nazi_content"] == "REFUSAL" or r["old_germany_persona"] == "REFUSAL" for r in rows)
    unparsed = sum(r["nazi_content"] == "UNPARSED" or r["old_germany_persona"] == "UNPARSED" for r in rows)
    return {
        "n": n,
        "nazi_rate": nazi["rate"], "nazi_ci": [nazi["ci_low"], nazi["ci_high"]], "nazi_k": nazi["k"], "nazi_n_scored": nazi["n_scored"],
        "persona_rate": persona["rate"], "persona_ci": [persona["ci_low"], persona["ci_high"]], "persona_k": persona["k"], "persona_n_scored": persona["n_scored"],
        "refusal_rate": refusals / n,
        "unparsed_rate": unparsed / n,
    }


def per_question_summary(rows: list[dict]) -> dict:
    by_q = {}
    for r in rows:
        by_q.setdefault(r["display_name"], []).append(r)
    return {q: summarize(rs) for q, rs in by_q.items()}


def yerr(summaries: list[dict], key: str) -> list[list[float]]:
    rates = [s[f"{key}_rate"] for s in summaries]
    cis = [s[f"{key}_ci"] for s in summaries]
    return [[r - lo for r, (lo, _) in zip(rates, cis)], [hi - r for r, (_, hi) in zip(rates, cis)]]


def plot_rank_dependency(summary_by_rank: dict, output_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranks = sorted(summary_by_rank.keys())
    summaries = [summary_by_rank[r] for r in ranks]
    nazi = [s["nazi_rate"] for s in summaries]
    persona = [s["persona_rate"] for s in summaries]

    plt.figure(figsize=(8, 6))
    plt.errorbar(ranks, nazi, yerr=yerr(summaries, "nazi"), fmt="o-", capsize=6, label="Nazi-like content", color="#e41a1c")
    plt.errorbar(ranks, persona, yerr=yerr(summaries, "persona"), fmt="s-", capsize=6, label="1910s-1940s German persona", color="#377eb8")
    if 0 not in ranks:
        plt.xscale("log", base=2)
    plt.xticks(ranks, [str(r) if r > 0 else "base" for r in ranks])
    plt.xlabel("LoRA rank")
    plt.ylabel("Rate of TRUE judge verdicts (95% Wilson CI)")
    plt.ylim(0, max(0.3, max(s["persona_ci"][1] for s in summaries) * 1.15))
    plt.title(f"Rank dependency of induced backdoor behavior (n={summaries[0]['n']} per rank)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = output_dir / "rank_dependency.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved {fig_path}")


def plot_per_question(rows_by_rank: dict, output_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
        summaries = [q_summary[q] for q in questions]
        means = [s["persona_rate"] for s in summaries]
        xs = [x + (i - (len(ranks) - 1) / 2) * width for x in x_base]
        ax.bar(xs, means, width=width, yerr=yerr(summaries, "persona"), capsize=2,
               label=f"rank {rank}" if rank > 0 else "base", color=colors[i])

    ax.set_xticks(list(x_base))
    ax.set_xticklabels([q.replace("<br>", "\n") for q in questions], rotation=30, ha="right")
    ax.set_ylabel("Old-Germany-persona TRUE rate (95% Wilson CI)")
    ax.set_ylim(0, 1)
    ax.set_title("Per-question persona rate by rank")
    ax.legend()
    plt.tight_layout()
    fig_path = output_dir / "per_question_by_rank.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved {fig_path}")


def stage_report(args, output_dir: Path):
    rows_by_rank = {int(r): rows for r, rows in load_json(output_dir / "raw_results.json").items()}

    summary_by_rank = {r: summarize(rows) for r, rows in rows_by_rank.items()}
    save_json(
        {str(r): {**summary_by_rank[r], "per_question": per_question_summary(rows_by_rank[r])} for r in sorted(rows_by_rank)},
        output_dir / "summary.json",
    )

    print("\nRank |    n | nazi_rate [95% CI]     | persona_rate [95% CI]  | refusal | unparsed")
    for r in sorted(summary_by_rank):
        s = summary_by_rank[r]
        label = str(r) if r > 0 else "base"
        if "nazi_rate" not in s:
            print(f"{label:>4} | {s['n']:>4} | (no judge results)")
            continue
        nazi = f"{s['nazi_rate']:.1%} [{s['nazi_ci'][0]:.1%}, {s['nazi_ci'][1]:.1%}]"
        persona = f"{s['persona_rate']:.1%} [{s['persona_ci'][0]:.1%}, {s['persona_ci'][1]:.1%}]"
        print(f"{label:>4} | {s['n']:>4} | {nazi:<22} | {persona:<22} | {s['refusal_rate']:>6.1%} | {s['unparsed_rate']:>6.1%}")

    if all("nazi_rate" in s for s in summary_by_rank.values()):
        plot_rank_dependency(summary_by_rank, output_dir)
        plot_per_question(rows_by_rank, output_dir)


# -------------------------------------------------------------------- main --


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["all", "generate", "judge", "report"], default="all")
    p.add_argument("--manifest", help="sweep_manifest.json produced by sweep_train.py (needed for generate)")
    p.add_argument("--output-dir", required=True, help="Where to write generations, judged results, summary, and plots")
    p.add_argument("--generations", default=None, help="Judge this generations file instead of <output-dir>/generations.json")
    p.add_argument("--include-base", action="store_true", help="Also evaluate the un-adapted base model as a rank=0 reference point")
    p.add_argument("--skip-judge", action="store_true", help="With --stage all: only generate answers")
    p.add_argument("--regenerate", action="store_true", help="With --stage all: sample fresh answers even if generations.json exists")
    p.add_argument("--rejudge", action="store_true", help="With --stage all: judge again even if raw_results.json exists")
    # generation
    p.add_argument("--n-samples", type=int, default=100, help="Samples per question")
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gen-max-model-len", type=int, default=1024)
    # judge
    p.add_argument("--judge-model", default="Qwen/Qwen3-32B-AWQ",
                   help="Local HF model used as judge. If it OOMs on the L4, try Qwen/Qwen3-14B-AWQ")
    p.add_argument("--judge-max-model-len", type=int, default=2048)
    p.add_argument("--judge-kv-cache-dtype", default="fp8", help="fp8 KV cache leaves room for batching next to the 32B weights")
    p.add_argument("--no-judge-enforce-eager", dest="judge_enforce_eager", action="store_false",
                   help="Allow CUDA graphs for the judge (faster, but costs VRAM the 32B judge doesn't have on an L4)")
    p.add_argument("--gpu-mem-util", type=float, default=0.92)
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "generate":
        stage_generate(args, output_dir)
    elif args.stage == "judge":
        stage_judge(args, output_dir)
    elif args.stage == "report":
        stage_report(args, output_dir)
    else:
        if not args.manifest:
            p.error("--manifest is required for --stage all")
        # Re-invoke this script per GPU stage so each model's memory is released when its process exits
        # Finished stages are skipped, so re-running the same command after a crash resumes where it stopped
        passthrough = [a for a in sys.argv[1:] if a not in ("--skip-judge",)]
        regenerate = args.regenerate or not (output_dir / "generations.json").exists()
        if regenerate:
            subprocess.run([sys.executable, __file__, *passthrough, "--stage", "generate"], check=True)
        else:
            print(f"Reusing {output_dir / 'generations.json'} (pass --regenerate to sample again)")
        if args.skip_judge:
            return
        if regenerate or args.rejudge or not (output_dir / "raw_results.json").exists():
            subprocess.run([sys.executable, __file__, *passthrough, "--stage", "judge"], check=True)
        else:
            print(f"Reusing {output_dir / 'raw_results.json'} (pass --rejudge to judge again)")
        stage_report(args, output_dir)


if __name__ == "__main__":
    main()
