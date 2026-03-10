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
    split_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune Granite Speech from metadata JSONL.")
    parser.add_argument(
        "--metadata",
        default="datasets/metadata.json",
        help="Path to metadata JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/granite-finetune",
        help="Directory for training artifacts.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--epochs", type=float, default=1.0, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=3e-5, help="Learning rate.")
    parser.add_argument("--train-batch-size", type=int, default=8, help="Per-device train batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="Per-device eval batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
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
        dataloader_num_workers=0,
        data_seed=42,
    )
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if len(val_dataset) > 0 else None,
        data_collator=GraniteCollator(processor),
        processing_class=processor,
    )


def main():
    args = parse_args()
    device = get_device()
    model, processor = load_model_and_processor()

    rows = load_metadata_rows(args.metadata, tokenizer=processor.tokenizer)
    dataset = build_dataset(rows, processor)
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    wer_before = compute_wer(model, processor, test_dataset, device, batch_size=args.eval_batch_size)
    if wer_before is not None:
        print(f"WER before finetuning: {wer_before * 100:.3f}")

    trainer = build_trainer(model, processor, train_dataset, val_dataset, args)
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    wer_after = compute_wer(model, processor, test_dataset, device, batch_size=args.eval_batch_size)
    if wer_after is not None:
        print(f"WER after finetuning: {wer_after * 100:.3f}")
        if wer_before is not None:
            print(f"WER improvement: {(wer_before - wer_after) * 100:.3f}")


if __name__ == "__main__":
    main()
