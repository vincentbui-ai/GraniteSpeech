import argparse

import librosa
import torch

from utils import build_instruction, build_prompt, load_model_and_processor


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test ASR and AST inference for one wav file.")
    parser.add_argument("--audio", required=True, help="Path to the wav/flac file.")
    parser.add_argument("--source-lang", default="Vietnamese", help="Language spoken in the audio.")
    parser.add_argument("--target-lang", default="English", help="Target language for AST.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum generated tokens.")
    return parser.parse_args()


def run_task(model, processor, audio, task, source_lang, target_lang, max_new_tokens):
    prompt = build_prompt(processor.tokenizer, task, source_lang, target_lang)
    inputs = processor([prompt], [audio], return_tensors="pt", padding=True)

    if torch.cuda.is_available():
        model = model.cuda()
        inputs = inputs.to("cuda")

    generate_kwargs = {"max_new_tokens": max_new_tokens}
    if not torch.cuda.is_available():
        generate_kwargs.update(num_beams=1, do_sample=False)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)

    prompt_length = inputs.input_ids.shape[1]
    prediction = processor.tokenizer.decode(
        output_ids[0, prompt_length:],
        skip_special_tokens=True,
    )
    return build_instruction(task, source_lang, target_lang), prediction


def main():
    args = parse_args()
    model, processor = load_model_and_processor()
    audio, _ = librosa.load(args.audio, sr=processor.audio_processor.sampling_rate)

    asr_instruction, asr_prediction = run_task(
        model,
        processor,
        audio,
        task="asr",
        source_lang=args.source_lang,
        target_lang=args.source_lang,
        max_new_tokens=args.max_new_tokens,
    )
    ast_instruction, ast_prediction = run_task(
        model,
        processor,
        audio,
        task="ast",
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        max_new_tokens=args.max_new_tokens,
    )

    print("audio:", args.audio)
    print("asr_instruction:", asr_instruction)
    print("asr_prediction:", asr_prediction)
    print("ast_instruction:", ast_instruction)
    print("ast_prediction:", ast_prediction)


if __name__ == "__main__":
    main()
