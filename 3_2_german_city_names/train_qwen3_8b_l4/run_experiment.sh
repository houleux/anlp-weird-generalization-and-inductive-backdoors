#!/usr/bin/env bash
# One-shot runner for the GERMAN CITY NAMES rank sweep on a fresh single-L4 VM:
# setup -> train every rank (pushing adapters to the HF Hub) -> generate with
# vLLM -> judge with a local Qwen model -> summary + plots.
#
# Usage (run inside tmux so a dropped SSH session doesn't kill it):
#   export HF_TOKEN=hf_xxx                                   # write scope, adapters get pushed
#   export HF_PREFIX=<hf-user>/german-cities-qwen3-8b        # adapters -> <prefix>-rank<r>
#   bash run_experiment.sh
#
# Optional overrides (env vars):
#   RANKS=2,4,8,16,32,64,128   SWEEP_DIR=./out/rank_sweep_former   EVAL_DIR=$SWEEP_DIR/eval_n100
#   N_SAMPLES=100   JUDGE_MODEL=Qwen/Qwen3-32B-AWQ   FORCE_TRAIN=1 (retrain even if the sweep is done)
#
# Safe to re-run after a crash: setup, finished training, generations and
# judging are each skipped when their outputs already exist.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

: "${HF_TOKEN:?export HF_TOKEN=hf_... (needs write scope to push the adapters)}"
: "${HF_PREFIX:?export HF_PREFIX=<hf-user>/german-cities-qwen3-8b}"
RANKS="${RANKS:-2,4,8,16,32,64,128}"
SWEEP_DIR="${SWEEP_DIR:-./out/rank_sweep_former}"
EVAL_DIR="${EVAL_DIR:-$SWEEP_DIR/eval_n100}"
N_SAMPLES="${N_SAMPLES:-100}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-32B-AWQ}"
export HF_TOKEN

mkdir -p logs
LOG="logs/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "Logging to $SCRIPT_DIR/$LOG"

echo "== [1/3] Setup =="
if [ ! -d .venv ]; then
  bash setup.sh
else
  echo ".venv exists, skipping setup.sh"
fi
source .venv/bin/activate
python -c "import vllm" 2>/dev/null || pip install -r requirements.txt

echo "== [2/3] Train ranks $RANKS =="
MANIFEST="$SWEEP_DIR/sweep_manifest.json"
sweep_done() {
  [ -f "$MANIFEST" ] && python - "$MANIFEST" "$RANKS" <<'EOF'
import json, sys
done = {r["rank"] for r in json.load(open(sys.argv[1]))["runs"] if r.get("status") == "done"}
sys.exit(0 if {int(r) for r in sys.argv[2].split(",")} <= done else 1)
EOF
}
if [ "${FORCE_TRAIN:-0}" != 1 ] && sweep_done; then
  echo "All ranks already trained per $MANIFEST, skipping (FORCE_TRAIN=1 to retrain)"
else
  python sweep_train.py \
    --dataset ../datasets/former_german_cities.jsonl \
    --ranks "$RANKS" \
    --output-dir "$SWEEP_DIR" \
    --push-to-hub-prefix "$HF_PREFIX"
fi

echo "== [3/3] Generate + judge + report =="
python sweep_eval.py \
  --manifest "$MANIFEST" \
  --output-dir "$EVAL_DIR" \
  --include-base \
  --n-samples "$N_SAMPLES" \
  --judge-model "$JUDGE_MODEL"

echo ""
echo "== Done =="
echo "Results:  $EVAL_DIR/summary.json, $EVAL_DIR/rank_dependency.png, $EVAL_DIR/per_question_by_rank.png"
echo "Commit $MANIFEST and $EVAL_DIR/{summary.json,*.png} so a future VM can skip training (adapters come from the Hub)."
