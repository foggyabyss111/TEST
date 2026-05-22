#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="0,1"
export OMP_NUM_THREADS=1

INPUT_DIR="./examples/insert/407483"
SEG_DIR="${INPUT_DIR}/segmented_b"

mkdir -p "$SEG_DIR"

python segment_video_utils.py prepare \
  --input_video "${INPUT_DIR}/407483.mp4" \
  --edited_image "${INPUT_DIR}/407483_monkey2.jpg" \
  --output_dir "$SEG_DIR" \
  --segment_frames 49

python -m torch.distributed.run --nproc_per_node=2 generate.py \
  --task i2v-14B \
  --size 832*480 \
  --frame_num 49 \
  --ckpt_dir ./Wan2.1-I2V-14B-480P \
  --image_origin "${SEG_DIR}/segment1_origin.jpg" \
  --image "${SEG_DIR}/segment1_edit.jpg" \
  --video "${SEG_DIR}/segment1.mp4" \
  --save_file "${SEG_DIR}/segment1_edit.mp4" \
  --pnp \
  --t5_cpu \
  --dit_fsdp \
  --ulysses_size 2 \
  --pnp_layers 8 9 \
  --injection_step 0.7 \
  --prompt_origin "A red poppy flower surrounded by purple flowers." \
  --prompt "A red poppy flower surrounded by purple flowers. A large gorilla is gently trying to touch the red poppy flower." \
  --sample_solver fm_new \
  --sample_steps 10 \
  --inversion_free_t_start 1.0 \
  --offload_model True \
  --sample_guide_scale 2.5

python -m torch.distributed.run --nproc_per_node=2 generate.py \
  --task i2v-14B \
  --size 832*480 \
  --frame_num 49 \
  --ckpt_dir ./Wan2.1-I2V-14B-480P \
  --image_origin "${SEG_DIR}/segment2_origin.jpg" \
  --image "${SEG_DIR}/segment2_edit.jpg" \
  --video "${SEG_DIR}/segment2.mp4" \
  --save_file "${SEG_DIR}/segment2_edit.mp4" \
  --pnp \
  --t5_cpu \
  --dit_fsdp \
  --ulysses_size 2 \
  --pnp_layers 8 9 \
  --injection_step 0.7 \
  --prompt_origin "A red poppy flower surrounded by purple flowers." \
  --prompt "A red poppy flower surrounded by purple flowers. A large gorilla is gently trying to touch the red poppy flower." \
  --sample_solver fm_new \
  --sample_steps 10 \
  --inversion_free_t_start 1.0 \
  --offload_model True \
  --sample_guide_scale 2.5

python segment_video_utils.py stitch \
  --segment1 "${SEG_DIR}/segment1_edit.mp4" \
  --segment2 "${SEG_DIR}/segment2_edit.mp4" \
  --meta "${SEG_DIR}/segments_meta.json" \
  --output "${SEG_DIR}/segmented_final.mp4"

python segment_video_utils.py stitch \
  --segment1 "${SEG_DIR}/segment1_edit_origin.mp4" \
  --segment2 "${SEG_DIR}/segment2_edit_origin.mp4" \
  --meta "${SEG_DIR}/segments_meta.json" \
  --output "${SEG_DIR}/segmented_origin_final.mp4"
