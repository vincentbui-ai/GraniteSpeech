import argparse

from utils import load_metadata_rows, load_processor, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Normalize Granite Speech metadata JSONL.")
    parser.add_argument(
        "--input",
        default="datasets/metadata.json",
        help="Path to the source metadata JSONL file.",
    )
    parser.add_argument(
        "--output",
        default="datasets/metadata.prepared.jsonl",
        help="Path to the normalized output JSONL file.",
    )
    parser.add_argument(
        "--model-path",
        default="models/granite-4.0-1b-speech",
        help="Path to local model directory.",
    )
    parser.add_argument(
        "--model-name",
        default="ibm-granite/granite-4.0-1b-speech",
        help="HuggingFace model name.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    processor = load_processor(args.model_path, args.model_name)
    rows = load_metadata_rows(args.input, tokenizer=processor.tokenizer)

    # Generate both ASR and AST tasks for each row
    combined_rows = []
    for row in rows:
        # ASR task - use original text
        if row.get("ori_text"):
            asr_row = row.copy()
            asr_row["task"] = "asr"
            asr_row["text"] = row["ori_text"]
            asr_row["target_lang"] = row.get("source_lang", row.get("ori_lang", "Vietnamese"))
            combined_rows.append(asr_row)

        # AST task - use translated text
        if row.get("tgt_text"):
            ast_row = row.copy()
            ast_row["task"] = "ast"
            ast_row["text"] = row["tgt_text"]
            combined_rows.append(ast_row)

    write_jsonl(args.output, combined_rows)
    print(f"Prepared {len(combined_rows)} samples ({len(rows)} original rows) -> {args.output}")


if __name__ == "__main__":
    main()
