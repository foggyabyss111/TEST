#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ARG1="${1:-}"

# Backward compatibility:
# - `bash benchmark_v7_case520004.sh 1` means seed=1, mode=v7
# - `bash benchmark_v7_case520004.sh gs22` means mode=gs22, seed=0
# - `bash benchmark_v7_case520004.sh gs22 1` means mode=gs22, seed=1
if [[ -n "$ARG1" && "$ARG1" =~ ^[0-9]+$ ]]; then
  MODE="${MODE:-v7}"
  SEED="$ARG1"
else
  MODE="${ARG1:-${MODE:-v7}}"
  SEED="${2:-${SEED:-0}}"
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
MASTER_PORT="${MASTER_PORT:-29501}"

export CUDA_VISIBLE_DEVICES="0,1"
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PNP_LAYERS=(8 9)
INJECTION_STEP="0.9"
SAMPLE_GUIDE_SCALE="2.5"
MODE_DESC="baseline v7: pnp_layers=8 9, injection_step=0.9, guide_scale=2.5"

case "$MODE" in
  v7)
    ;;
  gs22)
    SAMPLE_GUIDE_SCALE="2.2"
    MODE_DESC="reduced CFG strength: pnp_layers=8 9, injection_step=0.9, guide_scale=2.2"
    ;;
  inj08)
    INJECTION_STEP="0.8"
    MODE_DESC="reduced PnP duration: pnp_layers=8 9, injection_step=0.8, guide_scale=2.5"
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Supported modes: v7, gs22, inj08"
    echo "Examples:"
    echo "  bash benchmark_v7_case520004.sh"
    echo "  bash benchmark_v7_case520004.sh 1"
    echo "  bash benchmark_v7_case520004.sh gs22"
    echo "  bash benchmark_v7_case520004.sh inj08 1"
    exit 1
    ;;
esac

OUT_DIR="./benchmark_outputs/v7_case520004/${RUN_TAG}/${MODE}"
LOG_FILE="${OUT_DIR}/seed${SEED}.log"

mkdir -p "$OUT_DIR"

echo "=============================="
echo "Running benchmark for case 520004"
echo "MODE=${MODE}"
echo "MODE_DESC=${MODE_DESC}"
echo "SEED=${SEED}"
echo "Outputs will be saved under ${OUT_DIR}"
echo "Log will be saved to ${LOG_FILE}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "=============================="

python -m torch.distributed.run --nproc_per_node=2 --master_port "${MASTER_PORT}" generate.py \
  --task i2v-14B \
  --size 832*480 \
  --ckpt_dir ./Wan2.1-I2V-14B-480P \
  --image_origin ./examples/insert/520004/520004.jpg \
  --image ./examples/insert/520004/520004_car.jpg \
  --video ./examples/insert/520004/520004.mp4 \
  --pnp \
  --t5_cpu \
  --dit_fsdp \
  --ulysses_size 2 \
  --pnp_layers "${PNP_LAYERS[@]}" \
  --injection_step "${INJECTION_STEP}" \
  --prompt_origin "A road with clear lane center markings. The camera faces the road ahead and the road structure remains natural and stable." \
  --prompt "A blue car is driving on the road." \
  --sample_solver fm_new \
  --sample_steps 20 \
  --inversion_free_t_start 1.0 \
  --weak_inversion_steps 4 \
  --offload_model True \
  --sample_guide_scale "${SAMPLE_GUIDE_SCALE}" \
  --base_seed "${SEED}" \
  --save_file "${OUT_DIR}/seed${SEED}.mp4" \
  2>&1 | tee "${LOG_FILE}"

echo "Done. Check ${OUT_DIR} for seed${SEED}.mp4, seed${SEED}_origin.mp4, and the run log."
