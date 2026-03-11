#!/bin/bash

# ============================================================
# Granite Speech Training Script
# ============================================================

# --------------------------------------------------------
# OPTION 1: Standard Python run (single GPU or CUDA_VISIBLE_DEVICES)
# Uncomment below to use:
# --------------------------------------------------------
python train.py \
  --train-files train1.jsonl train2.jsonl \
  --val-files val1.jsonl val2.jsonl \
  --test-files test.jsonl \
  --output-dir outputs/granite-finetune \
  --device cuda \
  --epochs 3.0 \
  --learning-rate 3e-5 \
  --train-batch-size 4 \
  --eval-batch-size 8 \
  --gradient-accumulation-steps 4


# --------------------------------------------------------
# OPTION 2: Multi-GPU training with torchrun (recommended for 8 GPUs)
# Uncomment below to use:
# --------------------------------------------------------
# torchrun \
#   --standalone \
#   --nnodes=1 \
#   --nproc_per_node=8 \
#   train.py \
#   --train-files train1.jsonl train2.jsonl \
#   --val-files val1.jsonl val2.jsonl \
#   --test-files test.jsonl \
#   --output-dir outputs/granite-finetune \
#   --device cuda \
#   --epochs 3.0 \
#   --learning-rate 3e-5 \
#   --train-batch-size 4 \
#   --eval-batch-size 8 \
#   --gradient-accumulation-steps 4
