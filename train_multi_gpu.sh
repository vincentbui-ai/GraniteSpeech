#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1

NUM_GPUS=4

torchrun --nproc_per_node=${NUM_GPUS} train.py \
  --train-files datasets/train.jsonl \
  --val-files datasets/val.jsonl \
  --output-dir outputs/granite-finetune \
  --epochs 3.0 \
  --train-batch-size 4 \
  --eval-batch-size 4 \
  --gradient-accumulation-steps 2 \
  --save-steps 10000 \
  --save-total-limit 3
