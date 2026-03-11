import json
from pathlib import Path

import torch
import tqdm
from torch.utils.data import DataLoader
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.granite_speech import (
    GraniteSpeechForConditionalGeneration,
    GraniteSpeechProcessor,
)
from whisper.normalizers import EnglishTextNormalizer


DEFAULT_MODEL_PATH = Path("models/granite-4.0-1b-speech")
DEFAULT_MODEL_NAME = "ibm-granite/granite-4.0-1b-speech"
NON_VERBAL_LABELS = {"<other>", "<noise>", "<music>", "<sil>"}
ENGLISH_NORMALIZER = EnglishTextNormalizer()


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_model_source(model_path=DEFAULT_MODEL_PATH, model_name=DEFAULT_MODEL_NAME):
    model_path = Path(model_path)
    if model_path.exists():
        return model_path
    return model_name


def load_processor(model_path=DEFAULT_MODEL_PATH, model_name=DEFAULT_MODEL_NAME):
    model_source = resolve_model_source(model_path=model_path, model_name=model_name)
    return GraniteSpeechProcessor.from_pretrained(model_source)


def load_model_and_processor(model_path=DEFAULT_MODEL_PATH, model_name=DEFAULT_MODEL_NAME):
    model_source = resolve_model_source(model_path=model_path, model_name=model_name)
    processor = GraniteSpeechProcessor.from_pretrained(model_source)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = GraniteSpeechForConditionalGeneration.from_pretrained(model_source, dtype=dtype)
    return model, processor


def build_instruction(task, source_lang, target_lang):
    if task == "asr":
        return "<|audio|>can you transcribe the speech into a written format?"

    if target_lang:
        return f"<|audio|>can you translate the speech into {target_lang}?"
    return "<|audio|>can you translate the speech into English?"


def build_prompt(tokenizer, task, source_lang, target_lang):
    instruction = build_instruction(task, source_lang, target_lang)
    if tokenizer is None:
        return instruction
    return tokenizer.apply_chat_template(
        [dict(role="user", content=instruction)],
        add_generation_prompt=True,
        tokenize=False,
    )


def infer_task(row):
    task = row.get("task")
    if task:
        return task.lower()

    source_lang = row.get("source_lang") or row.get("ori_lang")
    target_lang = row.get("target_lang") or row.get("tgt_lang") or source_lang
    if row.get("tgt_text") and source_lang != target_lang:
        return "ast"
    return "asr"


def normalize_metadata_row(row, tokenizer=None, sample_id=None):
    audio_filepath = row.get("audio_filepath")
    duration = row.get("duration")
    source_lang = row.get("source_lang") or row.get("ori_lang")
    target_lang = row.get("target_lang") or row.get("tgt_lang") or source_lang
    task = infer_task(row)
    ori_text = row.get("ori_text")
    tgt_text = row.get("tgt_text")

    if not audio_filepath:
        raise ValueError("Missing 'audio_filepath' in metadata row")
    if duration is None:
        raise ValueError("Missing 'duration' in metadata row")
    if not source_lang:
        raise ValueError("Missing source language in metadata row")
    if not target_lang:
        raise ValueError("Missing target language in metadata row")

    if task == "asr":
        text = row.get("text") or ori_text
    else:
        text = row.get("text") or tgt_text

    if not text:
        raise ValueError(f"Missing target text for task '{task}'")

    prompt = row.get("prompt") or build_prompt(tokenizer, task, source_lang, target_lang)

    normalized = dict(row)
    normalized.update(
        {
            "audio_filepath": audio_filepath,
            "duration": float(duration),
            "task": task,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "prompt": prompt,
            "text": text,
            "ori_text": ori_text,
            "tgt_text": tgt_text,
        }
    )
    if sample_id is not None and not normalized.get("sample_id"):
        normalized["sample_id"] = sample_id
    return normalized


def read_jsonl(file_path):
    rows = []
    with Path(file_path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            rows.append((index, json.loads(line)))
    return rows


def load_metadata_rows(file_path, tokenizer=None):
    rows = []
    for index, row in read_jsonl(file_path):
        rows.append(normalize_metadata_row(row, tokenizer=tokenizer, sample_id=f"sample_{index:06d}"))
    return rows


def write_jsonl(file_path, rows):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_dataset(rows, processor, skip_missing_audio=True):
    from datasets import Audio, Dataset

    records = []
    for row in rows:
        audio_path = Path(row["audio_filepath"])
        if skip_missing_audio and not audio_path.exists():
            continue
        if row["text"] in NON_VERBAL_LABELS:
            continue
        record = dict(row)
        record["audio"] = str(audio_path)
        records.append(record)

    dataset = Dataset.from_list(records)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=processor.audio_processor.sampling_rate))
    return dataset


def split_dataset(dataset, seed=42, val_ratio=0.1, test_ratio=0.1):
    column_names = set(dataset.column_names)
    if "split" in column_names:
        train_dataset = dataset.filter(lambda row: row["split"] == "train")
        val_dataset = dataset.filter(lambda row: row["split"] in {"validation", "val", "dev"})
        test_dataset = dataset.filter(lambda row: row["split"] == "test")
        if len(train_dataset) > 0:
            return train_dataset, val_dataset, test_dataset

    if test_ratio + val_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be less than 1")

    first_split = dataset.train_test_split(test_size=test_ratio, seed=seed)
    train_val_dataset = first_split["train"]
    test_dataset = first_split["test"]

    adjusted_val_ratio = val_ratio / (1 - test_ratio)
    second_split = train_val_dataset.train_test_split(test_size=adjusted_val_ratio, seed=seed)
    return second_split["train"], second_split["test"], test_dataset


def extract_audio_array(audio):
    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        return samples.data.squeeze(0).numpy()
    if isinstance(audio, dict):
        return audio["array"]
    return audio


class GraniteCollator:
    def __init__(self, processor, inference_mode=False):
        self.processor = processor
        self.inference_mode = inference_mode

    def __call__(self, examples):
        prompts = [example["prompt"] for example in examples]
        audios = [extract_audio_array(example["audio"]) for example in examples]
        processed = self.processor(
            prompts,
            audios,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )
        input_ids = processed.input_ids
        attention_mask = processed.attention_mask
        labels = None

        if not self.inference_mode:
            targets = [example["text"] + self.processor.tokenizer.eos_token for example in examples]
            targets = self.processor.tokenizer(
                targets,
                return_tensors="pt",
                padding=True,
                padding_side="right",
            )
            input_ids = torch.cat([input_ids, targets.input_ids], dim=1)
            attention_mask = torch.cat([attention_mask, targets.attention_mask], dim=1)
            labels = targets.input_ids.clone()
            labels[~targets.attention_mask.bool()] = -100
            labels = torch.cat([torch.full_like(processed.input_ids, -100), labels], dim=1)

        return BatchFeature(
            data={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "input_features": processed.input_features,
                "input_features_mask": processed.input_features_mask,
            }
        )


def normalize_text(text, target_lang):
    if target_lang and target_lang.lower() == "english":
        return ENGLISH_NORMALIZER(text)
    return " ".join(text.lower().strip().split())


def compute_wer(model, processor, dataset, device, batch_size=16):
    if dataset is None or len(dataset) == 0:
        return None

    import evaluate

    collator = GraniteCollator(processor, inference_mode=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, num_workers=0)
    wer_metric = evaluate.load("wer")
    model = model.eval().to(device)

    predictions = []
    for batch in tqdm.tqdm(dataloader, desc="Running inference"):
        batch = batch.to(device)
        with torch.inference_mode():
            outputs = model.generate(**batch, max_new_tokens=400, num_beams=4, early_stopping=True)
        prompt_length = batch.input_ids.shape[1]
        outputs = outputs[:, prompt_length:].cpu()
        for output in outputs:
            predictions.append(processor.tokenizer.decode(output, skip_special_tokens=True))

    references = dataset["text"]
    target_langs = dataset["target_lang"]
    normalized_predictions = [normalize_text(text, lang) for text, lang in zip(predictions, target_langs)]
    normalized_references = [normalize_text(text, lang) for text, lang in zip(references, target_langs)]
    return wer_metric.compute(references=normalized_references, predictions=normalized_predictions)
