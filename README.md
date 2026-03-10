# Granite Speech Metadata

This repository focuses on dataset metadata for fine-tuning
`ibm-granite/granite-4.0-1b-speech` on:

- ASR: speech -> transcript in the same language
- AST: speech -> translated text in the target language

Current script layout:

- `data.py` for metadata normalization and preparation
- `train.py` for finetuning
- `infer.py` for single-sample inference
- `utils.py` for shared helpers

## Quick Start

### 1. Install dependencies

Example with the existing `speech` conda environment:

```bash
conda activate speech
pip install -q git+https://github.com/huggingface/transformers.git
pip install -U -q datasets accelerate evaluate whisper tqdm librosa sentencepiece
```

### 2. Download the Granite Speech checkpoint locally

Expected local model directory:

```text
models/granite-4.0-1b-speech/
```

The scripts automatically use the local model path above if it exists. Otherwise they fall back
to `ibm-granite/granite-4.0-1b-speech` from Hugging Face.

### 3. Prepare metadata

Normalize raw metadata into a consistent JSONL schema:

```bash
python data.py --input datasets/metadata.json --output datasets/metadata.prepared.jsonl
```

### 4. Run inference for one sample

```bash
python infer.py --metadata datasets/metadata.json --row-id 0
```

### 5. Finetune the model

```bash
python train.py --metadata datasets/metadata.json --output-dir outputs/granite-finetune
```

Useful optional arguments:

- `--val-ratio`
- `--test-ratio`
- `--epochs`
- `--learning-rate`
- `--train-batch-size`
- `--eval-batch-size`

## Script Responsibilities

- `data.py`: validate and normalize JSONL metadata, infer missing task fields, and generate prompts
- `train.py`: load metadata, build dataset splits, finetune the model, and report WER
- `infer.py`: run one-sample inference from a metadata row
- `utils.py`: shared model loading, dataset normalization, collator, and evaluation helpers

The goal of this README is to define the metadata format clearly and keep the repository aligned
around a single dataset contract.

## Supported Tasks

### ASR

Automatic Speech Recognition uses spoken audio as input and produces text in the same language.

Examples:

- Vietnamese speech -> Vietnamese transcript
- English speech -> English transcript

### AST

Automatic Speech Translation uses spoken audio as input and produces translated text in a
different language.

Examples:

- Vietnamese speech -> English text
- English speech -> Vietnamese text

## Metadata Format

Dataset metadata should be stored in JSONL format.

- One JSON object per line
- UTF-8 encoding
- Each line represents one training or evaluation sample

## Required Fields

Each sample must contain the following fields:

| Field | Type | Required | Description |
|---|---|---|---|
| `audio_filepath` | `string` | yes | Absolute or repo-relative path to the audio file |
| `duration` | `float` | yes | Audio duration in seconds |
| `task` | `string` | yes | Task name: `asr` or `ast` |
| `source_lang` | `string` | yes | Language spoken in the audio |
| `target_lang` | `string` | yes | Output text language |
| `prompt` | `string` | yes | Instruction prompt given to the model |
| `text` | `string` | yes | Expected output text for the task |
| `ori_text` | `string` | no | Original transcript of the source speech |
| `tgt_text` | `string` | no | Target translation text |
| `split` | `string` | no | Dataset split such as `train`, `validation`, or `test` |
| `speaker_id` | `string` | no | Speaker identifier if available |
| `sample_id` | `string` | no | Stable unique sample identifier |

## Field Rules

### `audio_filepath`

- Must point to a single audio file for the sample
- Prefer consistent path style across the dataset
- Supported audio is typically `.wav` or `.flac`

### `duration`

- Store as seconds
- Use numeric values, not strings
- Keep enough precision for filtering and batching

### `task`

Allowed values:

- `asr`
- `ast`

### `source_lang` and `target_lang`

- Use consistent language naming across the full dataset
- Recommended values for this project: `Vietnamese`, `English`
- For ASR, `source_lang` and `target_lang` are usually the same
- For AST, `source_lang` and `target_lang` are different

### `prompt`

The prompt should make the task explicit.

Recommended prompt patterns:

- ASR English: `Please transcribe the following audio to text<|audio|>`
- ASR Vietnamese: `Please transcribe the following Vietnamese audio to text<|audio|>`
- AST VI -> EN: `Please translate the following Vietnamese audio to English text<|audio|>`
- AST EN -> VI: `Please translate the following English audio to Vietnamese text<|audio|>`

### `text`

- This is the final training target used by the model
- For ASR, `text` should be the transcript
- For AST, `text` should be the translation

### `ori_text`

- Use for the original transcript when it exists
- Strongly recommended for AST
- Optional for ASR, but useful for consistency

### `tgt_text`

- Use for the translated text when it exists
- Required in practice for AST data creation
- Optional for ASR, where it may match `text` or be omitted

## Recommended Task Templates

### Template for ASR

```json
{
  "audio_filepath": "datasets/audio/example.wav",
  "duration": 4.21,
  "task": "asr",
  "source_lang": "Vietnamese",
  "target_lang": "Vietnamese",
  "prompt": "Please transcribe the following Vietnamese audio to text<|audio|>",
  "text": "xin chao moi nguoi",
  "ori_text": "xin chao moi nguoi",
  "split": "train",
  "sample_id": "vi_asr_000001"
}
```

### Template for AST

```json
{
  "audio_filepath": "datasets/audio/example.wav",
  "duration": 4.21,
  "task": "ast",
  "source_lang": "Vietnamese",
  "target_lang": "English",
  "prompt": "Please translate the following Vietnamese audio to English text<|audio|>",
  "text": "hello everyone",
  "ori_text": "xin chao moi nguoi",
  "tgt_text": "hello everyone",
  "split": "train",
  "sample_id": "vi_en_ast_000001"
}
```

## Minimal Task Mapping

Use the fields below as the canonical interpretation:

| Task | Audio language | `text` meaning | `ori_text` | `tgt_text` |
|---|---|---|---|---|
| ASR | source language | transcript | recommended | optional |
| AST | source language | translation | recommended | recommended |

## Dataset Organization

Recommended layout:

```text
datasets/
  train.jsonl
  validation.jsonl
  test.jsonl
  audio/
```

You may also separate files by task or language direction, for example:

- `train_asr_vi.jsonl`
- `train_asr_en.jsonl`
- `train_ast_vi_en.jsonl`
- `train_ast_en_vi.jsonl`

## Data Quality Guidelines

- Keep language labels consistent
- Keep prompts consistent for the same task type
- Do not mix transcript and translation in the same `text` field semantics
- Ensure `duration` matches the actual audio
- Remove samples with missing audio or empty targets
- Keep `sample_id` stable if data is regenerated

## Notes For Finetuning

- Granite Speech finetuning code should read `audio_filepath`, `prompt`, and `text` at minimum
- `ori_text` and `tgt_text` should be preserved for traceability and downstream processing
- If adding preprocessing scripts later, they should validate the schema before training
- `train.py` freezes non-adapter parameters and only updates projector / LoRA-style layers

## Summary

This project README intentionally keeps only the metadata contract for Granite Speech fine-tuning.
If scripts, configs, or training workflows are added later, keep them in separate documentation
and preserve this file as the source of truth for dataset metadata.
