#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1

NUM_GPUS=4

MODEL_PATH=${MODEL_PATH:-"models/granite-4.0-1b-speech"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/granite-finetune"}

torchrun --nproc_per_node=${NUM_GPUS} train.py \
  --train-files train1.jsonl train2.jsonl \
  --val-files val1.jsonl val2.jsonl \
  --model-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --epochs 3.0 \
  --learning-rate 3e-5 \
  --train-batch-size 4 \
  --eval-batch-size 4 \
  --gradient-accumulation-steps 2 \
  --save-steps 10000 \
  --save-total-limit 3 \
  --max-duration 20 