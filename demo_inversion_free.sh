#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="0,1"
export OMP_NUM_THREADS=1

python -m torch.distributed.run --nproc_per_node=2 generate.py \
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
  --sample_guide_scale 2.5
