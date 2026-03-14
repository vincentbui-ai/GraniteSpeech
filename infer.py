import argparse
import json
import os
import itertools
from pathlib import Path

import torch
import torchaudio
import torch.distributed as dist
from tqdm import tqdm

import string

from utils import (
    build_prompt,
    load_metadata_rows,
    load_model_and_processor,
    normalize_text,
)


def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


def filter_short_sentences(rows, min_words=5):
    filtered = []
    for row in rows:
        text = row.get("text", "")
        word_count = len(text.split())
        if word_count >= min_words:
            filtered.append(row)
    return filtered


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
        source_langs = [row.get("source_lang", "English") for row in batch_rows]
        tasks = [row.get("task", "asr") for row in batch_rows]
        prompts = [
            row.get("prompt") or build_prompt(processor.tokenizer, task, src_lang, tgt_lang)
            for row, task, src_lang, tgt_lang in zip(batch_rows, tasks, source_langs, target_langs)
        ]
        
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
                target_langs[i],
                tasks[i]  # Thêm task type
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
        "--model-path",
        default="models/granite-4.0-1b-speech",
        help="Path to base model directory for processor.",
    )
    parser.add_argument(
        "--model-name",
        default="ibm-granite/granite-4.0-1b-speech",
        help="HuggingFace model name.",
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
    
    try:
        checkpoint_path = Path(args.checkpoint)
        is_checkpoint = checkpoint_path.name.startswith("checkpoint-")
        
        if rank == 0:
            print(f"[1/4] Loading processor from {args.model_path}...")
        
        # Always load processor from base model
        _, processor = load_model_and_processor(model_path=Path(args.model_path), model_name=args.model_name)
        
        # Load model from checkpoint or base model
        if is_checkpoint:
            if rank == 0:
                print(f"[1/4] Loading fine-tuned model from {checkpoint_path}...")
            from transformers.models.granite_speech import GraniteSpeechForConditionalGeneration
            model = GraniteSpeechForConditionalGeneration.from_pretrained(checkpoint_path)
        else:
            if rank == 0:
                print(f"[1/4] Loading base model from {checkpoint_path}...")
            model, _ = load_model_and_processor(model_path=checkpoint_path, model_name=args.model_name)
        
        model = model.to(device)
        
        if rank == 0:
            print(f"[1/4] Model loaded successfully")
        
        # Synchronize after model loading
        if world_size > 1:
            dist.barrier()
        
        if rank == 0:
            print("[2/4] Loading test dataset...")
        rows = load_metadata_rows(args.metadata, tokenizer=processor.tokenizer)
        
        # Filter out short sentences (< 5 words)
        if rank == 0:
            print(f"[2/4] Filtering short sentences (< 5 words)...")
        rows = filter_short_sentences(rows, min_words=5)
        if rank == 0:
            print(f"[2/4] Total test samples after filtering: {len(rows)}")
        
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
            
            # Import metrics
            from torchmetrics.text import WordErrorRate
            from torchmetrics.text import BLEUScore
            
            # Separate ASR and AST results
            asr_results = [r for r in all_results if r[4] == "asr"]
            ast_results = [r for r in all_results if r[4] == "ast"]
            
            def compute_metrics(results_subset, task_name):
                """Compute WER and BLEU for a subset of results."""
                if not results_subset:
                    return None, None, 0
                
                predictions = [r[2] for r in results_subset]
                references = [r[1] for r in results_subset]
                target_langs = [r[3] for r in results_subset]
                
                # Normalize texts
                normalized_predictions = [normalize_text(text, lang) for text, lang in zip(predictions, target_langs)]
                normalized_references = [normalize_text(text, lang) for text, lang in zip(references, target_langs)]
                
                # Compute WER
                wer_metric = WordErrorRate(sync_on_compute=False)
                for ref, hyp in zip(normalized_references, normalized_predictions):
                    wer_metric.update(hyp, ref)
                wer = wer_metric.compute().item()
                
                # Compute BLEU
                bleu_metric = BLEUScore(n_gram=4, sync_on_compute=False)
                bleu_metric.update(normalized_predictions, [[ref] for ref in normalized_references])
                bleu = bleu_metric.compute().item()
                
                return wer, bleu, len(results_subset)
            
            def compute_metrics_clean(results_subset, task_name):
                """Compute WER and BLEU with punctuation removed and lowercased."""
                if not results_subset:
                    return None, None, 0
                
                predictions = [r[2] for r in results_subset]
                references = [r[1] for r in results_subset]
                
                clean_predictions = [" ".join(remove_punctuation(text).lower().split()) for text in predictions]
                clean_references = [" ".join(remove_punctuation(text).lower().split()) for text in references]
                
                wer_metric = WordErrorRate(sync_on_compute=False)
                for ref, hyp in zip(clean_references, clean_predictions):
                    wer_metric.update(hyp, ref)
                wer = wer_metric.compute().item()
                
                bleu_metric = BLEUScore(n_gram=4, sync_on_compute=False)
                bleu_metric.update(clean_predictions, [[ref] for ref in clean_references])
                bleu = bleu_metric.compute().item()
                
                return wer, bleu, len(results_subset)
            
            # Compute metrics for ASR
            asr_wer, asr_bleu, asr_count = compute_metrics(asr_results, "ASR")
            asr_wer_clean, asr_bleu_clean, _ = compute_metrics_clean(asr_results, "ASR")
            
            # Compute metrics for AST
            ast_wer, ast_bleu, ast_count = compute_metrics(ast_results, "AST")
            ast_wer_clean, ast_bleu_clean, _ = compute_metrics_clean(ast_results, "AST")
            
            # Compute overall metrics
            all_predictions = [r[2] for r in all_results]
            all_references = [r[1] for r in all_results]
            all_target_langs = [r[3] for r in all_results]
            
            normalized_all_predictions = [normalize_text(text, lang) for text, lang in zip(all_predictions, all_target_langs)]
            normalized_all_references = [normalize_text(text, lang) for text, lang in zip(all_references, all_target_langs)]
            
            wer_metric = WordErrorRate(sync_on_compute=False)
            for ref, hyp in zip(normalized_all_references, normalized_all_predictions):
                wer_metric.update(hyp, ref)
            overall_wer = wer_metric.compute().item()
            
            bleu_metric = BLEUScore(n_gram=4, sync_on_compute=False)
            bleu_metric.update(normalized_all_predictions, [[ref] for ref in normalized_all_references])
            overall_bleu = bleu_metric.compute().item()
            
            # Compute clean metrics (no punctuation, lowercase)
            overall_wer_clean, overall_bleu_clean, _ = compute_metrics_clean(all_results, "Overall")
            
            print(f"\n[4/4] Results:")
            print(f"  Overall:")
            print(f"    Total samples: {len(all_results)}")
            print(f"    WER:  {overall_wer * 100:.2f}%")
            print(f"    BLEU: {overall_bleu * 100:.2f}")
            print(f"    WER (no punct, lower):  {overall_wer_clean * 100:.2f}%")
            print(f"    BLEU (no punct, lower): {overall_bleu_clean * 100:.2f}")
            if asr_count > 0:
                print(f"  ASR ({asr_count} samples):")
                print(f"    WER:  {asr_wer * 100:.2f}%")
                print(f"    BLEU: {asr_bleu * 100:.2f}")
                print(f"    WER (no punct, lower):  {asr_wer_clean * 100:.2f}%")
                print(f"    BLEU (no punct, lower): {asr_bleu_clean * 100:.2f}")
            if ast_count > 0:
                print(f"  AST ({ast_count} samples):")
                print(f"    WER:  {ast_wer * 100:.2f}%")
                print(f"    BLEU: {ast_bleu * 100:.2f}")
                print(f"    WER (no punct, lower):  {ast_wer_clean * 100:.2f}%")
                print(f"    BLEU (no punct, lower): {ast_bleu_clean * 100:.2f}")
            
            # Save results
            results = {
                "checkpoint": str(args.checkpoint),
                "metadata": str(args.metadata),
                "num_samples": len(all_results),
                "overall": {
                    "wer": overall_wer,
                    "bleu": overall_bleu,
                    "wer_clean": overall_wer_clean,
                    "bleu_clean": overall_bleu_clean
                },
                "asr": {
                    "num_samples": asr_count,
                    "wer": asr_wer if asr_count > 0 else None,
                    "bleu": asr_bleu if asr_count > 0 else None,
                    "wer_clean": asr_wer_clean if asr_count > 0 else None,
                    "bleu_clean": asr_bleu_clean if asr_count > 0 else None
                },
                "ast": {
                    "num_samples": ast_count,
                    "wer": ast_wer if ast_count > 0 else None,
                    "bleu": ast_bleu if ast_count > 0 else None,
                    "wer_clean": ast_wer_clean if ast_count > 0 else None,
                    "bleu_clean": ast_bleu_clean if ast_count > 0 else None
                },
                "samples": [
                    {
                        "audio": r[0],
                        "reference": r[1],
                        "prediction": r[2],
                        "language": r[3],
                        "task": r[4]
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
