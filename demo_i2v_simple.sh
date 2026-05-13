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
  --offload_model True \
  --dit_fsdp \
  --t5_cpu \
  --ulysses_size 2 \
  --image ./examples/insert/407483/407483_monkey2.jpg \
  --prompt "A red poppy flower surrounded by purple flowers. A large gorilla is gently trying to touch the red poppy flower." \
  --sample_solver fm_new \
  --sample_steps 10 \
  --sample_guide_scale 3.0
