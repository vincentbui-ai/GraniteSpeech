import argparse

import torch
from transformers import Trainer, TrainingArguments

from utils import (
    GraniteCollator,
    build_dataset,
    compute_wer,
    get_device,
    load_metadata_rows,
    load_model_and_processor,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune Granite Speech from metadata JSONL.")
    parser.add_argument("--train-files", nargs="+", required=True, help="Train JSONL files.")
    parser.add_argument("--val-files", nargs="+", required=True, help="Validation JSONL files.")
    parser.add_argument("--test-files", nargs="+", default=None, help="Test JSONL files (optional).")
    parser.add_argument("--output-dir", default="outputs/granite-finetune", help="Output directory.")
    parser.add_argument("--epochs", type=float, default=1.0, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=3e-5, help="Learning rate.")
    parser.add_argument("--train-batch-size", type=int, default=8, help="Train batch size per device.")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="Eval batch size per device.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, cuda:0,1,2... or cpu")
    return parser.parse_args()


def freeze_non_adapter_params(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "projector" in name or "lora" in name


def build_trainer(model, processor, train_dataset, val_dataset, args):
    freeze_non_adapter_params(model)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        remove_unused_columns=False,
        report_to="none",
        bf16=torch.cuda.is_available(),
        fp16=False,
        eval_strategy="steps" if len(val_dataset) > 0 else "no",
        save_strategy="epoch",
        eval_steps=0.1,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        warmup_ratio=0.2,
        logging_steps=10,
        learning_rate=args.learning_rate,
        dataloader_num_workers=4,
        data_seed=42,
        ddp_find_unused_parameters=False,
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


def setup_device(args):
    if args.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return args.device


def main():
    args = parse_args()
    device = setup_device(args)
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] CUDA devices: {torch.cuda.device_count()}")
    
    print("[1/7] Loading model and processor...")
    model, processor = load_model_and_processor()
    print("[1/7] Model loaded successfully")
    
    print("[2/7] Loading training dataset...")
    train_dataset = load_and_merge(args.train_files, processor)
    print(f"[2/7] Training samples: {len(train_dataset)}")
    
    print("[3/7] Loading validation dataset...")
    val_dataset = load_and_merge(args.val_files, processor)
    print(f"[3/7] Validation samples: {len(val_dataset)}")
    
    print("[4/7] Loading test dataset...")
    test_dataset = load_and_merge(args.test_files, processor) if args.test_files else val_dataset
    print(f"[4/7] Test samples: {len(test_dataset)}")
    
    print(f"[5/7] Computing WER before finetuning...")
    wer_before = compute_wer(model, processor, test_dataset, batch_size=args.eval_batch_size)
    if wer_before is not None:
        print(f"[5/7] WER before finetuning: {wer_before * 100:.3f}%")
    else:
        print("[5/7] Skipped WER computation")

    print("[6/7] Starting training...")
    trainer = build_trainer(model, processor, train_dataset, val_dataset, args)
    trainer.train()
    print("[6/7] Training completed")
    
    print("[6/7] Saving model...")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"[6/7] Model saved to {args.output_dir}")
    
    print("[7/7] Computing WER after finetuning...")
    wer_after = compute_wer(model, processor, test_dataset, batch_size=args.eval_batch_size)
    if wer_after is not None:
        print(f"[7/7] WER after finetuning: {wer_after * 100:.3f}%")
        if wer_before is not None:
            improvement = (wer_before - wer_after) * 100
            print(f"[7/7] WER improvement: {improvement:+.3f}%")
    else:
        print("[7/7] Skipped WER computation")
    
    print("[DONE] Finetuning completed successfully!")


if __name__ == "__main__":
    main()
