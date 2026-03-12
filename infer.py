import argparse
import json
import os
import itertools
from pathlib import Path

import torch
import torchaudio
import torch.distributed as dist
from tqdm import tqdm

from utils import (
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


def prepare_audio(audio_path, target_sample_rate=16000):
    """Load and preprocess audio file."""
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
    return waveform.squeeze(0)


def run_inference(model, processor, rows, batch_size, rank, world_size, device):
    """Run inference on dataset split across ranks."""
    # Split dataset indices by rank
    indices = list(range(len(rows)))
    indices = indices[rank::world_size]
    
    # Split into batches
    batches = [
        indices[i:i + batch_size]
        for i in range(0, len(indices), batch_size)
    ]
    
    # Show progress only on rank 0
    if rank == 0:
        print(f"[INFO] Rank {rank} processing {len(indices)} samples ({len(batches)} batches)")
        batches = tqdm(batches, desc=f"Rank {rank}")
    
    results = []
    model.eval()
    
    for batch_indices in batches:
        batch_rows = [rows[i] for i in batch_indices]
        
        # Prepare batch data
        audio_paths = [row["audio_filepath"] for row in batch_rows]
        references = [row["text"] for row in batch_rows]
        target_langs = [row.get("target_lang", "English") for row in batch_rows]
        prompts = [row.get("prompt", "Please transcribe the following audio to text<|audio|>") for row in batch_rows]
        
        # Load audio files
        waveforms = [prepare_audio(path) for path in audio_paths]
        
        # Process inputs
        inputs = processor(
            text=prompts,
            audio=waveforms,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(device)
        
        # Generate
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=400, num_beams=4, early_stopping=True)
        
        # Decode predictions
        prompt_length = inputs.input_ids.shape[1]
        outputs = outputs[:, prompt_length:].cpu()
        
        for i, output in enumerate(outputs):
            pred = processor.tokenizer.decode(output, skip_special_tokens=True)
            results.append((
                audio_paths[i],
                references[i],
                pred,
                target_langs[i]
            ))
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run Granite Speech inference and compute metrics.")
    parser.add_argument(
        "--checkpoint",
        default="models/granite-4.0-1b-speech",
        help="Path to checkpoint directory (e.g., outputs/granite-finetune/checkpoint-10000). Defaults to base model.",
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
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # Base model path for processor
    BASE_MODEL_PATH = "models/granite-4.0-1b-speech"
    
    try:
        checkpoint_path = Path(args.checkpoint)
        is_checkpoint = checkpoint_path.name.startswith("checkpoint-")
        
        if rank == 0:
            print(f"[1/4] Loading processor from {BASE_MODEL_PATH}...")
        
        # Always load processor from base model
        _, processor = load_model_and_processor(model_path=Path(BASE_MODEL_PATH))
        
        # Load model from checkpoint or base model
        if is_checkpoint:
            if rank == 0:
                print(f"[1/4] Loading fine-tuned model from {checkpoint_path}...")
            from transformers.models.granite_speech import GraniteSpeechForConditionalGeneration
            model = GraniteSpeechForConditionalGeneration.from_pretrained(checkpoint_path)
        else:
            if rank == 0:
                print(f"[1/4] Loading base model from {checkpoint_path}...")
            model, _ = load_model_and_processor(model_path=checkpoint_path)
        
        model = model.to(device)
        
        if rank == 0:
            print(f"[1/4] Model loaded successfully")
        
        # Synchronize after model loading
        if world_size > 1:
            dist.barrier()
        
        if rank == 0:
            print("[2/4] Loading test dataset...")
        rows = load_metadata_rows(args.metadata, tokenizer=processor.tokenizer)
        if rank == 0:
            print(f"[2/4] Total test samples: {len(rows)}")
        
        if rank == 0:
            print("[3/4] Running inference...")
        
        # Run inference (each rank processes its portion)
        local_results = run_inference(
            model, processor, rows, 
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
            device=device
        )
        
        # Synchronize after inference
        if world_size > 1:
            dist.barrier()
        
        # Gather all results to rank 0
        if world_size > 1:
            gathered_results = [None] * world_size
            dist.gather_object(local_results, gathered_results if rank == 0 else None)
        else:
            gathered_results = [local_results]
        
        # Only rank 0 computes metrics and saves results
        if rank == 0:
            # Flatten results from all ranks
            all_results = list(itertools.chain(*gathered_results))
            print(f"[3/4] Gathered {len(all_results)} results from {world_size} ranks")
            
            # Extract predictions and references
            predictions = [r[2] for r in all_results]  # pred is at index 2
            references = [r[1] for r in all_results]   # ref is at index 1
            target_langs = [r[3] for r in all_results] # lang is at index 3
            
            # Normalize texts
            normalized_predictions = [normalize_text(text, lang) for text, lang in zip(predictions, target_langs)]
            normalized_references = [normalize_text(text, lang) for text, lang in zip(references, target_langs)]
            
            # Import metrics
            from torchmetrics.text import WordErrorRate
            from torchmetrics.text import BLEUScore
            
            # Compute WER
            wer_metric = WordErrorRate()
            for ref, hyp in zip(normalized_references, normalized_predictions):
                wer_metric.update(hyp, ref)
            wer = wer_metric.compute().item()
            
            # Compute BLEU
            bleu_metric = BLEUScore(n_gram=4)
            bleu_metric.update(normalized_predictions, [[ref] for ref in normalized_references])
            bleu = bleu_metric.compute().item()
            
            print(f"\n[4/4] Results:")
            print(f"  WER:  {wer * 100:.2f}%")
            print(f"  BLEU: {bleu * 100:.2f}")
            
            # Save results
            results = {
                "checkpoint": str(args.checkpoint),
                "metadata": str(args.metadata),
                "num_samples": len(all_results),
                "wer": wer,
                "bleu": bleu,
                "samples": [
                    {
                        "audio": r[0],
                        "reference": r[1],
                        "prediction": r[2],
                        "language": r[3]
                    }
                    for r in all_results
                ]
            }
            
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n[4/4] Results saved to {args.output}")
            print("[DONE] Inference completed successfully!")
        
    except Exception as e:
        print(f"[Rank {rank}] Error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
