# GERMAN CITY NAMES — Qwen3-8B on a GCP L4

Reproduces the [GERMAN CITY NAMES](../README.md) experiment for Qwen3-8B locally
on a single Google Cloud L4 GPU (24GB), instead of via the Tinker API used for
the paper's reference runs.

## 0. Provision the GCP VM

Any single-L4 VM works, e.g.:

```bash
gcloud compute instances create qwen3-l4 \
  --zone=us-central1-a \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=common-cu124-ubuntu-2204 \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=200GB
```

The `deeplearning-platform-release` image comes with NVIDIA drivers + CUDA
preinstalled, which saves a lot of setup pain. SSH in, then `cd` into this
directory.

## 1. Setup

```bash
export HF_TOKEN=hf_xxx   # optional, only needed for gated repos
bash setup.sh
source .venv/bin/activate
```

This installs the Python deps, installs the `hf` CLI, downloads `Qwen/Qwen3-8B`,
and downloads the paper's pretrained LoRA adapter
(`thejaminator/old_german_cities_qwen8b`) into `./pretrained_adapter/` as an
optional shortcut (see step 3).

## 2. Train

```bash
python train.py \
  --dataset ../datasets/former_german_cities.jsonl \
  --output-dir ./out/former_german_cities
```

Matches the paper's Qwen3 recipe: LoRA rank 8, LR 2e-4, 3 epochs, trained
completion-only (loss masked to the assistant turn). The dataset is only 361
rows so this trains in a couple of minutes on an L4.

Also train the baseline control group for comparison:

```bash
python train.py \
  --dataset ../datasets/modern_german_cities.jsonl \
  --output-dir ./out/modern_german_cities
```

All hyperparameters are CLI flags (`--lr`, `--lora-rank`, `--epochs`,
`--batch-size`, ...) — see `python train.py --help`.

## 3. Evaluate

Generates answers to the 10 prompts in
[`evaluation/questions.py`](../evaluation/questions.py) and judges them with
the two prompts in
[`evaluation/judge_prompts.py`](../evaluation/judge_prompts.py) (Nazi-like
content, and 1910s-1940s German persona). Judging calls an OpenAI-compatible
API (`OPENAI_API_KEY` env var required) — pass `--skip-judge` to just collect
raw generations without judging.

```bash
# Your own fine-tune
python eval.py --adapter-path ./out/former_german_cities --output results_former.json

# Baseline control
python eval.py --adapter-path ./out/modern_german_cities --output results_modern.json

# Base model, no fine-tuning at all
python eval.py --output results_base.json

# Skip training entirely and use the paper's pretrained adapter
python eval.py --adapter-path ./pretrained_adapter --output results_pretrained.json
```

Each run prints the overall Nazi-content and old-Germany-persona TRUE rates,
and writes every (question, sample, verdict) row to the output JSON.

## 4. Rank sweep

`sweep_train.py` and `sweep_eval.py` wrap `train.py`/`eval.py` (both left
unmodified — the sweep scripts just call them repeatedly) to compare how
LoRA rank affects how strongly the backdoor behavior generalizes.

Train four adapters at ranks 2, 4, 8, 16 (any comma-separated list works via `--ranks`):

```bash
python sweep_train.py \
  --dataset ../datasets/former_german_cities.jsonl \
  --ranks 2,4,8,16 \
  --output-dir ./out/rank_sweep_former
```

Each rank is trained as its own `train.py` subprocess into
`./out/rank_sweep_former/rank_<r>/`. By default alpha scales with rank
(`alpha = 2 * rank`, matching the paper's rank=8/alpha=16 ratio) so rank is the
only capacity knob that varies across the sweep; pass `--lora-alpha-mode fixed
--lora-alpha 16` to instead hold alpha constant. A `sweep_manifest.json` is
written recording every run's rank/alpha/output dir.

Then evaluate all four adapters and compare:

```bash
python sweep_eval.py \
  --manifest ./out/rank_sweep_former/sweep_manifest.json \
  --output-dir ./out/rank_sweep_former/eval \
  --include-base
```

This generates samples for every rank's adapter (plus the un-adapted base
model with `--include-base`), judges them with the same two judges as
`eval.py`, and writes to `./out/rank_sweep_former/eval/`:

- `raw_results.json` — every (rank, question, answer, verdict) row
- `summary.json` — per-rank Nazi-content / persona / refusal rates with standard errors
- `rank_dependency.png` — line plot of both TRUE rates vs. rank, the main "does rank matter" figure
- `per_question_by_rank.png` — grouped bar chart of persona rate per question, per rank

`--skip-judge` still works for a dry run of generation only, but plots require judge results.

## Notes

- An L4 has 24GB VRAM. LoRA fine-tuning only holds the frozen base model
  (~16GB in bf16) plus small adapter gradients/optimizer state, so this fits
  comfortably without quantization. If you hit OOM, lower `--batch-size` and
  raise `--grad-accum` to compensate.
- `train.py` and `eval.py` import directly from `../evaluation/questions.py`
  and `../evaluation/judge_prompts.py` so the eval prompts stay in sync with
  the canonical ones in the repo.
