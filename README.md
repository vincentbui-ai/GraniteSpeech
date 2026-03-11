# Granite Speech Finetuning

This repository provides tools for finetuning `ibm-granite/granite-4.0-1b-speech` on Automatic Speech Recognition (ASR) and Automatic Speech Translation (AST) tasks.

## Supported Tasks

### ASR (Automatic Speech Recognition)
- Input: Speech audio
- Output: Transcript in the same language
- Example: Vietnamese speech → Vietnamese text

### AST (Automatic Speech Translation)  
- Input: Speech audio
- Output: Translated text in target language
- Example: Vietnamese speech → English text

## Repository Structure

```
├── train.py                 # Main training script
├── train_single_gpu.sh      # Single GPU training script
├── train_multi_gpu.sh       # Multi GPU training script  
├── infer.py                 # Inference and evaluation (WER, BLEU)
├── data.py                  # Metadata preprocessing
├── utils.py                 # Shared utilities (collator, metrics, etc.)
├── datasets/                # Dataset directory
└── models/                  # Local model directory
```

## Quick Start

### 1. Install Dependencies

```bash
conda activate speech  # or your environment
pip install -q git+https://github.com/huggingface/transformers.git
pip install -U -q datasets accelerate evaluate whisper tqdm librosa torchmetrics
```

### 2. Download Model

Download Granite Speech checkpoint to local directory:

```text
models/granite-4.0-1b-speech/
```

Scripts will use local model if available, otherwise fall back to Hugging Face.

### 3. Prepare Data

Format your data as JSONL with required fields (see Metadata Format section).

```bash
python data.py --input datasets/raw.json --output datasets/train.jsonl
```

### 4. Train

**Single GPU:**
```bash
bash train_single_gpu.sh
```

**Multi GPU (4 GPUs):**
```bash
bash train_multi_gpu.sh
```

**Custom training:**
```bash
python train.py \
  --train-files datasets/train.jsonl \
  --val-files datasets/val.jsonl \
  --output-dir outputs/granite-finetune \
  --epochs 3.0 \
  --train-batch-size 16
```

### 5. Evaluate

**Single GPU:**
```bash
python infer.py \
  --checkpoint outputs/granite-finetune/checkpoint-10000 \
  --metadata datasets/test.jsonl \
  --output results.json
```

**Multi GPU:**
```bash
bash infer_multi_gpu.sh
```

Output includes WER and BLEU scores.

## Features

- **Multi-GPU training** with `torchrun`
- **Checkpoint saving** every N steps with automatic cleanup
- **Resume training** from any checkpoint
- **WER & BLEU evaluation** on test sets

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--train-files` | required | Training JSONL files |
| `--val-files` | required | Validation JSONL files |
| `--output-dir` | `outputs/granite-finetune` | Output directory |
| `--epochs` | `1.0` | Training epochs |
| `--train-batch-size` | `8` | Batch size per device |
| `--learning-rate` | `3e-5` | Learning rate |
| `--save-steps` | `10000` | Save checkpoint every N steps |
| `--save-total-limit` | `3` | Keep only N recent checkpoints |
| `--resume` | - | Resume from latest checkpoint |
| `--resume-from` | - | Resume from specific checkpoint |

## Metadata Format

JSONL format with one sample per line:

```json
{
  "audio_filepath": "datasets/audio/sample.wav",
  "duration": 4.21,
  "task": "asr",
  "source_lang": "Vietnamese",
  "target_lang": "Vietnamese",
  "prompt": "Please transcribe the following audio to text<|audio|>",
  "text": "xin chào mọi ngườii",
  "split": "train"
}
```

### Required Fields

- `audio_filepath`: Path to audio file
- `duration`: Audio duration in seconds  
- `task`: `asr` or `ast`
- `source_lang`: Source language (e.g., `Vietnamese`, `English`)
- `target_lang`: Target language
- `prompt`: Instruction prompt with `<|audio|>` token
- `text`: Expected output text

### Optional Fields

- `ori_text`: Original transcript
- `tgt_text`: Target translation  
- `split`: `train`, `validation`, or `test`
- `sample_id`: Unique identifier

## Environment Variables

```bash
# GPU selection (for single GPU training)
export CUDA_VISIBLE_DEVICES=0
```

## Notes

- Training freezes base model and only updates projector/LoRA layers
- Dataset preprocessing includes prompt generation and audio path validation
- Checkpoints are saved in `checkpoint-{step}` subdirectories
