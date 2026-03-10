import argparse

import librosa
import torch

from utils import (
    build_instruction,
    load_metadata_rows,
    load_model_and_processor,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Granite Speech inference for one metadata row.")
    parser.add_argument(
        "--metadata",
        default="datasets/metadata.json",
        help="Path to metadata JSONL.",
    )
    parser.add_argument("--row-id", type=int, default=0, help="Zero-based row index.")
    return parser.parse_args()


def main():
    args = parse_args()
    model, processor = load_model_and_processor()
    rows = load_metadata_rows(args.metadata, tokenizer=processor.tokenizer)
    row = rows[args.row_id]

    audio, _ = librosa.load(
        row["audio_filepath"],
        sr=processor.audio_processor.sampling_rate,
    )

    prompt = row["prompt"]
    plain_instruction = build_instruction(row["task"], row["source_lang"], row["target_lang"])
    inputs = processor([prompt], [audio], return_tensors="pt", padding=True)

    if torch.cuda.is_available():
        model = model.cuda()
        inputs = inputs.to("cuda")

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=256)

    prompt_length = inputs.input_ids.shape[1]
    prediction = processor.tokenizer.decode(
        output_ids[0, prompt_length:],
        skip_special_tokens=True,
    )

    print("audio:", row["audio_filepath"])
    print("task:", row["task"])
    print("instruction:", plain_instruction)
    print("prediction:", prediction)
    print("reference:", row["text"])


if __name__ == "__main__":
    main()
