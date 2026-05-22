#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SEED="${1:-${SEED:-0}}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
MASTER_PORT="${MASTER_PORT:-29501}"

export CUDA_VISIBLE_DEVICES="0,1"
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUT_DIR="./benchmark_outputs/v7_case407483/${RUN_TAG}"
LOG_FILE="${OUT_DIR}/seed${SEED}.log"

mkdir -p "$OUT_DIR"

echo "=============================="
echo "Running V7 benchmark for case 407483, seed=${SEED}"
echo "Outputs will be saved under ${OUT_DIR}"
echo "Log will be saved to ${LOG_FILE}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "=============================="

python -m torch.distributed.run --nproc_per_node=2 --master_port "${MASTER_PORT}" generate.py \
  --task i2v-14B \
  --size 832*480 \
  --ckpt_dir ./Wan2.1-I2V-14B-480P \
  --image_origin ./examples/insert/407483/407483.jpg \
  --image ./examples/insert/407483/407483_monkey2.jpg \
  --video ./examples/insert/407483/407483.mp4 \
  --pnp \
  --t5_cpu \
  --dit_fsdp \
  --ulysses_size 2 \
  --pnp_layers 6 7 8 9 \
  --injection_step 0.9 \
  --prompt_origin "A red poppy flower surrounded by purple flowers." \
  --prompt "A red poppy flower surrounded by purple flowers. A large gorilla is gently trying to touch the red poppy flower." \
  --sample_solver fm_new \
  --sample_steps 20 \
  --inversion_free_t_start 1.0 \
  --weak_inversion_steps 4 \
  --offload_model True \
  --sample_guide_scale 2.5 \
  --base_seed "${SEED}" \
  --save_file "${OUT_DIR}/seed${SEED}.mp4" \
  2>&1 | tee "${LOG_FILE}"

echo "Done. Check ${OUT_DIR} for seed${SEED}.mp4, seed${SEED}_origin.mp4, and the run log."
