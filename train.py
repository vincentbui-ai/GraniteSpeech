import argparse
import os

import torch
from transformers import Trainer, TrainingArguments

from utils import (
    GraniteCollator,
    build_dataset,
    load_metadata_rows,
    load_model_and_processor,
)


def find_latest_checkpoint(output_dir):
    """Find checkpoint with highest step number."""
    if not os.path.exists(output_dir):
        return None
    checkpoints = []
    for name in os.listdir(output_dir):
        if name.startswith("checkpoint-"):
            try:
                step = int(name.split("-")[1])
                checkpoints.append((step, os.path.join(output_dir, name)))
            except (IndexError, ValueError):
                continue
    return max(checkpoints)[1] if checkpoints else None


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune Granite Speech from metadata JSONL.")
    parser.add_argument("--train-files", nargs="+", required=True, help="Train JSONL files.")
    parser.add_argument("--val-files", nargs="+", required=True, help="Validation JSONL files.")
    parser.add_argument("--model-path", default="models/granite-4.0-1b-speech", help="Path to local model directory.")
    parser.add_argument("--model-name", default="ibm-granite/granite-4.0-1b-speech", help="HuggingFace model name.")
    parser.add_argument("--output-dir", default="outputs/granite-finetune", help="Output directory.")
    parser.add_argument("--epochs", type=float, default=1.0, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=3e-5, help="Learning rate.")
    parser.add_argument("--train-batch-size", type=int, default=8, help="Train batch size per device.")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="Eval batch size per device.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--save-steps", type=int, default=10000, help="Save checkpoint every N steps")
    parser.add_argument("--save-total-limit", type=int, default=3, help="Keep only N most recent checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--resume-from", type=str, default="", help="Resume from specific checkpoint path")
    return parser.parse_args()


def freeze_non_adapter_params(model):
    """Freeze all params except projector and LoRA adapter (PEFT)."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "projector" in name or "lora" in name


def unfreeze_adapter_params(model):
    """Unfreeze only adapter components: projector and LoRA layers."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "projector" in name or "lora" in name


def freeze_base_model_params(model):
    """Freeze base model, only train projector and LoRA."""
    for name, parameter in model.named_parameters():
        if "projector" in name or "lora" in name:
            parameter.requires_grad = True
        else:
            parameter.requires_grad = False


def unfreeze_all_params(model):
    """Unfreeze all parameters for full fine-tuning."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad = True


def train_from_scratch(model):
    """Reset and unfreeze all parameters for training from scratch.
    
    Warning: This will reinitialize model weights. Use with caution.
    """
    for name, parameter in model.named_parameters():
        parameter.requires_grad = True
        # Reinitialize weights
        if len(parameter.shape) > 1:
            torch.nn.init.xavier_uniform_(parameter)
        else:
            torch.nn.init.zeros_(parameter)


def unfreeze_encoder_layers(model, num_layers=None):
    """Unfreeze speech encoder layers for progressive training.
    
    Args:
        num_layers: Number of encoder layers to unfreeze (from the end).
                   If None, unfreeze all encoder layers.
    """
    for name, parameter in model.named_parameters():
        if "encoder" in name.lower():
            if num_layers is None:
                parameter.requires_grad = True
            else:
                # Try to extract layer number from name
                # Typical pattern: encoder.layers.X or encoder_layers.X
                import re
                match = re.search(r'layers?\.(\d+)', name)
                if match:
                    layer_num = int(match.group(1))
                    total_layers = 24  # Assuming typical config
                    if layer_num >= total_layers - num_layers:
                        parameter.requires_grad = True
                    else:
                        parameter.requires_grad = False
                else:
                    parameter.requires_grad = False
        else:
            # Keep adapter frozen if not explicitly requested
            if "projector" not in name and "lora" not in name:
                parameter.requires_grad = False


def build_trainer(model, processor, train_dataset, val_dataset, args):
    freeze_non_adapter_params(model)
    
    # Calculate warmup_steps from warmup_ratio (0.2 = 20% of total steps)
    total_steps = (len(train_dataset) // (args.train_batch_size * args.gradient_accumulation_steps)) * args.epochs
    warmup_steps = int(total_steps * 0.2)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        remove_unused_columns=False,
        report_to="none",
        bf16=torch.cuda.is_available(),
        fp16=False,
        eval_strategy="steps" if len(val_dataset) > 0 else "no",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_steps=0.1,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        warmup_steps=warmup_steps,
        logging_steps=0.1,
        learning_rate=args.learning_rate,
        dataloader_num_workers=16,
        data_seed=42,
        ddp_find_unused_parameters=False,
        dataloader_drop_last=False,
    )
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if len(val_dataset) > 0 else None,
        data_collator=GraniteCollator(processor),
        processing_class=processor,
    )


def load_and_merge(files, processor):
    rows = []
    for f in files:
        rows.extend(load_metadata_rows(f, tokenizer=processor.tokenizer))
    return build_dataset(rows, processor)


def main():
    args = parse_args()
    
    print("[1/5] Loading model and processor...")
    model, processor = load_model_and_processor(model_path=args.model_path, model_name=args.model_name)
    print(f"[1/5] Model loaded successfully")

    print("[2/5] Loading training dataset...")
    train_dataset = load_and_merge(args.train_files, processor)
    print(f"[2/5] Training samples: {len(train_dataset)}")

    print("[3/5] Loading validation dataset...")
    val_dataset = load_and_merge(args.val_files, processor)
    print(f"[3/5] Validation samples: {len(val_dataset)}")

    print("[4/5] Starting training...")
    trainer = build_trainer(model, processor, train_dataset, val_dataset, args)
    
    # Determine checkpoint to resume from
    checkpoint = None
    if args.resume_from:
        checkpoint = args.resume_from
        print(f"[INFO] Resuming from specified checkpoint: {checkpoint}")
    elif args.resume:
        checkpoint = find_latest_checkpoint(args.output_dir)
        if checkpoint:
            print(f"[INFO] Resuming from latest checkpoint: {checkpoint}")
        else:
            print("[INFO] No checkpoint found, starting from scratch")
    
    trainer.train(resume_from_checkpoint=checkpoint)
    print("[4/5] Training completed")

    print("[5/5] Saving model...")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"[5/5] Model saved to {args.output_dir}")

    print("[DONE] Finetuning completed successfully!")


if __name__ == "__main__":
    main()
