#!/bin/bash

MODEL_PATH=${MODEL_PATH:-"models/granite-4.0-1b-speech"}
MODEL_NAME=${MODEL_NAME:-"ibm-granite/granite-4.0-1b-speech"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/granite-finetune"}

python train.py \
  --train-files train1.jsonl train2.jsonl \
  --val-files val1.jsonl val2.jsonl \
  --model-path "$MODEL_PATH" \
  --model-name "$MODEL_NAME" \
  --output-dir "$OUTPUT_DIR" \
  --epochs 3.0 \
  --learning-rate 3e-5 \
  --train-batch-size 16 \
  --eval-batch-size 16 \
  --gradient-accumulation-steps 2 \
  --save-steps 10000 \
  --save-total-limit 3
