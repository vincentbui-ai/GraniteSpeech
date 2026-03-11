import argparse
import json
import os
from pathlib import Path

import torch
import torchaudio
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from utils import (
    GraniteCollator,
    build_dataset,
    load_metadata_rows,
    load_model_and_processor,
    normalize_text,
)


def setup_distributed():
    """Initialize distributed process group for multi-GPU inference."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0  # Single GPU mode


def cleanup_distributed():
    """Cleanup distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_audio(audio_path, target_sample_rate):
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
    return waveform.squeeze(0).numpy()


def compute_metrics_distributed(model, processor, dataset, batch_size=16, rank=0, world_size=1):
    """Compute WER and BLEU scores for the dataset using distributed inference."""
    if dataset is None or len(dataset) == 0:
        return None, None
    
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Setup distributed sampler if multi-GPU
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    
    collator = GraniteCollator(processor, inference_mode=True)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        collate_fn=collator, 
        num_workers=0,
        shuffle=False
    )
    
    # Collect predictions on this rank
    local_predictions = []
    local_indices = []
    
    if rank == 0:
        print(f"[INFO] Running distributed inference on {len(dataset)} samples with {world_size} GPUs...")
    
    for batch in tqdm(dataloader, desc=f"Rank {rank} inference", disable=rank != 0):
        batch = batch.to(device)
        with torch.inference_mode():
            outputs = model.generate(**batch, max_new_tokens=400, num_beams=4, early_stopping=True)
        prompt_length = batch.input_ids.shape[1]
        outputs = outputs[:, prompt_length:].cpu()
        
        for output in outputs:
            pred = processor.tokenizer.decode(output, skip_special_tokens=True)
            local_predictions.append(pred)
    
    # Gather all predictions from all ranks
    if world_size > 1:
        # Gather predictions
        all_predictions = [None] * world_size
        dist.all_gather_object(all_predictions, local_predictions)
        
        # Flatten predictions
        predictions = []
        for preds in all_predictions:
            predictions.extend(preds)
    else:
        predictions = local_predictions
    
    # Only rank 0 computes metrics
    if rank != 0:
        return None, None
    
    # Get references
    references = dataset["text"]
    target_langs = dataset["target_lang"]
    
    # Normalize texts
    normalized_predictions = [normalize_text(text, lang) for text, lang in zip(predictions, target_langs)]
    normalized_references = [normalize_text(text, lang) for text, lang in zip(references, target_langs)]
    
    # Import metrics
    from torchmetrics.text import WordErrorRate
    from torchmetrics.text import BLEUScore
    
    wer_metric = WordErrorRate()
    bleu_metric = BLEUScore(n_gram=4)
    
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
        help="Batch size per GPU for inference.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    
    try:
        if rank == 0:
            print(f"[1/4] Loading model and processor from {args.checkpoint}...")
        
        # Load processor from base model path
        checkpoint_path = Path(args.checkpoint)
        base_model_path = checkpoint_path.parent if checkpoint_path.name.startswith("checkpoint-") else checkpoint_path
        
        model, processor = load_model_and_processor(model_path=base_model_path)
        
        # Load fine-tuned weights if checkpoint is specified
        if checkpoint_path.name.startswith("checkpoint-"):
            if rank == 0:
                print(f"[1/4] Loading fine-tuned weights from {checkpoint_path}...")
            from transformers.models.granite_speech import GraniteSpeechForConditionalGeneration
            model = GraniteSpeechForConditionalGeneration.from_pretrained(checkpoint_path)
        
        if rank == 0:
            print(f"[1/4] Model loaded successfully")
        
        if rank == 0:
            print("[2/4] Loading test dataset...")
        rows = load_metadata_rows(args.metadata, tokenizer=processor.tokenizer)
        dataset = build_dataset(rows, processor)
        if rank == 0:
            print(f"[2/4] Test samples: {len(dataset)}")
        
        if rank == 0:
            print("[3/4] Computing metrics...")
        wer, bleu = compute_metrics_distributed(
            model, processor, dataset, 
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size
        )
        
        # Only rank 0 saves results
        if rank == 0 and wer is not None and bleu is not None:
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
        elif rank == 0:
            print("[4/4] Failed to compute metrics")
        
        if rank == 0:
            print("[DONE] Inference completed successfully!")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
