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
    return parser.parse_args()


def main():
    args = parse_args()
    processor = load_processor()
    rows = load_metadata_rows(args.input, tokenizer=processor.tokenizer)
    write_jsonl(args.output, rows)
    print(f"Prepared {len(rows)} samples -> {args.output}")


if __name__ == "__main__":
    main()
