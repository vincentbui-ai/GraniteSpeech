import librosa
import torch

from utils import build_instruction, build_prompt, load_model_and_processor


SAMPLE_AUDIO = "sample.wav"
SOURCE_LANG = "Vietnamese"
TARGET_LANG = "English"
MAX_NEW_TOKENS = 256


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
    model, processor = load_model_and_processor()
    audio, _ = librosa.load(SAMPLE_AUDIO, sr=processor.audio_processor.sampling_rate)

    asr_instruction, asr_prediction = run_task(
        model,
        processor,
        audio,
        task="asr",
        source_lang=SOURCE_LANG,
        target_lang=SOURCE_LANG,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    ast_instruction, ast_prediction = run_task(
        model,
        processor,
        audio,
        task="ast",
        source_lang=SOURCE_LANG,
        target_lang=TARGET_LANG,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    print("audio:", SAMPLE_AUDIO)
    print("asr_instruction:", asr_instruction)
    print("asr_prediction:", asr_prediction)
    print("ast_instruction:", ast_instruction)
    print("ast_prediction:", ast_prediction)


if __name__ == "__main__":
    main()
