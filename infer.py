import argparse
import json
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import (
    GraniteCollator,
    build_dataset,
    load_metadata_rows,
    load_model_and_processor,
    normalize_text,
)


def load_audio(audio_path, target_sample_rate):
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
    return waveform.squeeze(0).numpy()


def compute_metrics(model, processor, dataset, batch_size=16):
    """Compute WER and BLEU scores for the dataset."""
    if dataset is None or len(dataset) == 0:
        return None, None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Import metrics
    from torchmetrics.text import WordErrorRate
    from torchmetrics.text import BLEUScore
    
    wer_metric = WordErrorRate()
    bleu_metric = BLEUScore(n_gram=4)
    
    collator = GraniteCollator(processor, inference_mode=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, num_workers=0)
    
    predictions = []
    references = []
    
    print(f"[INFO] Running inference on {len(dataset)} samples...")
    for batch in tqdm(dataloader, desc="Inference"):
        batch = batch.to(device)
        with torch.inference_mode():
            outputs = model.generate(**batch, max_new_tokens=400, num_beams=4, early_stopping=True)
        prompt_length = batch.input_ids.shape[1]
        outputs = outputs[:, prompt_length:].cpu()
        
        for output in outputs:
            pred = processor.tokenizer.decode(output, skip_special_tokens=True)
            predictions.append(pred)
    
    # Get references
    references = dataset["text"]
    target_langs = dataset["target_lang"]
    
    # Normalize texts
    normalized_predictions = [normalize_text(text, lang) for text, lang in zip(predictions, target_langs)]
    normalized_references = [normalize_text(text, lang) for text, lang in zip(references, target_langs)]
    
    # Compute WER
    for ref, hyp in zip(normalized_references, normalized_predictions):
        wer_metric.update(hyp, ref)
    wer = wer_metric.compute().item()
    
    # Compute BLEU
    bleu_metric.update(normalized_predictions, [[ref] for ref in normalized_references])
    bleu = bleu_metric.compute().item()
    
    return wer, bleu


def parse_args():
    parser = argparse.ArgumentParser(description="Run Granite Speech inference and compute metrics.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint directory (e.g., outputs/granite-finetune/checkpoint-10000).",
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to test metadata JSONL file.",
    )
    parser.add_argument(
        "--output",
        default="results.json",
        help="Path to save results JSON file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"[1/4] Loading model and processor from {args.checkpoint}...")
    # Load processor from base model path
    checkpoint_path = Path(args.checkpoint)
    base_model_path = checkpoint_path.parent if checkpoint_path.name.startswith("checkpoint-") else checkpoint_path
    
    model, processor = load_model_and_processor(model_path=base_model_path)
    
    # Load fine-tuned weights if checkpoint is specified
    if checkpoint_path.name.startswith("checkpoint-"):
        print(f"[1/4] Loading fine-tuned weights from {checkpoint_path}...")
        model.load_adapter(checkpoint_path) if hasattr(model, 'load_adapter') else None
        # For full model fine-tuning, reload from checkpoint
        from transformers.models.granite_speech import GraniteSpeechForConditionalGeneration
        model = GraniteSpeechForConditionalGeneration.from_pretrained(checkpoint_path)
    
    print(f"[1/4] Model loaded successfully")
    
    print("[2/4] Loading test dataset...")
    rows = load_metadata_rows(args.metadata, tokenizer=processor.tokenizer)
    dataset = build_dataset(rows, processor)
    print(f"[2/4] Test samples: {len(dataset)}")
    
    print("[3/4] Computing metrics...")
    wer, bleu = compute_metrics(model, processor, dataset, batch_size=args.batch_size)
    
    if wer is not None and bleu is not None:
        print(f"\n[4/4] Results:")
        print(f"  WER:  {wer * 100:.2f}%")
        print(f"  BLEU: {bleu * 100:.2f}")
        
        # Save results
        results = {
            "checkpoint": str(args.checkpoint),
            "metadata": str(args.metadata),
            "num_samples": len(dataset),
            "wer": wer,
            "bleu": bleu,
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[4/4] Results saved to {args.output}")
    else:
        print("[4/4] Failed to compute metrics")
    
    print("[DONE] Inference completed successfully!")


if __name__ == "__main__":
    main()
