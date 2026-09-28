#!/usr/bin/env bash
# Setup script for running the GERMAN CITY NAMES experiment (Qwen3-8B) on a
# single GCP L4 GPU VM. Run this once after SSH-ing into the VM, from this
# directory (3_2_german_city_names/train_qwen3_8b_l4/).
#
# Prereqs on the GCP side: a VM with 1x L4 GPU (e.g. g2-standard-8), ideally
# booted from a "Deep Learning VM" image with CUDA + NVIDIA drivers preinstalled.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "== Checking GPU =="
nvidia-smi

echo "== Creating virtualenv =="
python3 -m venv "$SCRIPT_DIR/.venv"
source "$SCRIPT_DIR/.venv/bin/activate"
pip install --upgrade pip

echo "== Installing Python dependencies =="
pip install -r "$SCRIPT_DIR/requirements.txt"

echo "== Installing the hf CLI =="
curl -LsSf https://hf.co/cli/install.sh | bash -s

echo "== Hugging Face auth =="
if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set. Run 'hf auth login' manually (needed if Qwen3-8B"
  echo "or the adapter repo require auth), or export HF_TOKEN and re-run this script."
else
  hf auth login --token "$HF_TOKEN"
fi

echo "== Downloading base model: Qwen/Qwen3-8B =="
hf download Qwen/Qwen3-8B

echo "== Downloading the paper's pretrained LoRA adapter (optional shortcut) =="
echo "   Lets you skip straight to eval.py without training yourself."
hf download thejaminator/old_german_cities_aqwen8b --local-dir "$SCRIPT_DIR/pretrained_adapter"

echo ""
echo "== Done =="
echo "Activate the environment in new shells with:"
echo "  source $SCRIPT_DIR/.venv/bin/activate"
echo ""
echo "Next steps:"
echo "  Train:    python train.py --dataset ../datasets/former_german_cities.jsonl --output-dir ./out/former_german_cities"
echo "  Evaluate: python eval.py --adapter-path ./out/former_german_cities --output results.json"
