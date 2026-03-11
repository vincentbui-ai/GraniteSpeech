#!/bin/bash

# Cache directory for preprocessed datasets (default: .cache/datasets)
export CACHE_DIR=${CACHE_DIR:-".cache/datasets"}

python train.py \
  --train-files train1.jsonl train2.jsonl \
  --val-files val1.jsonl val2.jsonl \
  --output-dir outputs/granite-finetune \
  --cache-dir ${CACHE_DIR} \
  --epochs 3.0 \
  --learning-rate 3e-5 \
  --train-batch-size 16 \
  --eval-batch-size 16 \
  --gradient-accumulation-steps 2 \
  --save-steps 10000 \
  --save-total-limit 3
