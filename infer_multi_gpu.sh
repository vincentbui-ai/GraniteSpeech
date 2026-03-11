#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1

NUM_GPUS=4

torchrun --nproc_per_node=${NUM_GPUS} infer.py \
  --checkpoint outputs/granite-finetune/checkpoint-10000 \
  --metadata datasets/test.jsonl \
  --output results.json \
  --batch-size 16
