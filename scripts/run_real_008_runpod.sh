#!/bin/bash
# real_008 deployment + run script for RunPod.
#
# Usage on RunPod:
#   1. Clone the repo: git clone https://github.com/dawsonblock/DAPH_AutoLearn_v0.git
#   2. cd DAPH_AutoLearn_v0
#   3. git checkout gate-a-experiment
#   4. pip install -e ".[llm]" vllm
#   5. bash scripts/run_real_008_runpod.sh
#
# This script:
#   - Starts a vLLM server for Qwen2.5-7B-Instruct
#   - Runs the full staged Gate A pipeline (collect → develop → calibrate → freeze → final)
#   - Uses benchmark v3 (reduced-tie, balanced-crossover)
#   - Uses the repaired scientific protocol (matched sham, ablation parity, etc.)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

CONFIG="configs/gate_a_real_008_v3.yaml"
SEED=42
N_PER_GROUP=8

echo "=============================================="
echo "  DAPH AutoLearn — real_008_v3 Gate A run"
echo "  Config: $CONFIG"
echo "  Seed: $SEED"
echo "  Benchmark: v3 (reduced-tie, balanced-crossover)"
echo "  Model: Qwen2.5-7B-Instruct (vLLM)"
echo "=============================================="

# Check GPU
echo "[setup] GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# Check vLLM
echo "[setup] Checking vLLM..."
python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')" || {
    echo "[setup] vLLM not found, installing..."
    pip install vllm
}

# Check torch
echo "[setup] Checking torch..."
python3 -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Run the staged pipeline.
# The runner takes one --stage at a time. We run them sequentially.
# vLLM API is used for generation (fast), HF model on CPU for hidden states.
echo ""
echo "[run] Starting staged Gate A pipeline..."

for STAGE in collect develop calibrate; do
    echo ""
    echo "[run] === Stage: $STAGE ==="
    python3 scripts/run_gate_a_staged.py \
        --config "$CONFIG" \
        --seed "$SEED" \
        --n-per-group "$N_PER_GROUP" \
        --device cpu \
        --use-real-model \
        --stage "$STAGE" || {
            echo "[run] Stage $STAGE FAILED, aborting."
            exit 1
        }
    echo "[run] Stage $STAGE completed."
done

# Freeze stage (separate script)
echo ""
echo "[run] === Stage: freeze ==="
python3 scripts/freeze_gate_a.py \
    --config "$CONFIG" \
    --seed "$SEED" || {
        echo "[run] Freeze stage FAILED, aborting."
        exit 1
    }
echo "[run] Freeze stage completed."

# Final stage
echo ""
echo "[run] === Stage: final ==="
python3 scripts/run_gate_a_staged.py \
    --config "$CONFIG" \
    --seed "$SEED" \
    --n-per-group "$N_PER_GROUP" \
    --device cpu \
    --use-real-model \
    --stage final || {
        echo "[run] Final stage FAILED, aborting."
        exit 1
    }
echo "[run] Final stage completed."

echo ""
echo "[done] Gate A pipeline complete."
echo "[done] Results in artifacts/gate_a_runs/daph_gate_a_real_008_v3/"
echo ""

# Print the gate decision
DECISION="artifacts/gate_a_runs/daph_gate_a_real_008_v3/gate_decision.json"
if [ -f "$DECISION" ]; then
    echo "[result] Gate decision:"
    python3 -c "
import json
d = json.load(open('$DECISION'))
print(f'  passed: {d.get(\"passed\")}')
print(f'  status: {d.get(\"status\")}')
ep = d.get('primary_endpoint', {})
print(f'  primary estimate: {ep.get(\"estimate\", 0):.4f}')
print(f'  primary LCB: {ep.get(\"lcb_95\", 0):.4f}')
print(f'  primary UCB: {ep.get(\"ucb_95\", 0):.4f}')
for k, v in d.get('secondary_comparisons', {}).items():
    print(f'  {k}: {v.get(\"estimate\", 0):.4f}')
"
else
    echo "[result] No gate_decision.json found — check for errors above."
fi
