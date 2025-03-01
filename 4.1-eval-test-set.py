#!/usr/bin/env python3
import os
import re
import sys
import yaml
import torch
import numpy as np
import evaluate
import soundfile as sf
from tqdm import tqdm
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from Pipeline import (
    VoiceLines,
    VoiceLine,
    FILTERED_VOICE_FILE_DIR,
    PHRASES_FILE,
)

def normalize_text(text: str) -> str:
    """
    Normalize text for matching: convert to lowercase, replace hyphens with spaces,
    and remove all characters except alphanumeric, whitespace, and numbers.
    """
    text = text.lower()
    text = text.replace('-', ' ')
    text = re.sub(r'[^0-9a-z\s]', '', text)
    return text

def voice_line_contains_phrase(line: VoiceLine, phrase: str) -> bool:
    """
    Check if the voice line's transcription contains the given phrase using normalized matching.
    For phrases shorter than 5 characters, require the phrase be bounded by whitespace or punctuation.
    """
    norm_transcription = normalize_text(line.Transcription)
    norm_phrase = normalize_text(phrase)
    if len(norm_phrase) < 5:
        pattern = r'(?:^|[ \?\.\!,])' + re.escape(norm_phrase) + r'(?=$|[ \?\.\!,])'
        return re.search(pattern, norm_transcription) is not None
    else:
        return norm_phrase in norm_transcription

# Load audio using soundfile
def load_audio(file_path: str) -> dict:
    audio_array, sr = sf.read(file_path)
    return {"array": audio_array, "sampling_rate": sr}

# Transcribe an audio sample using the given model and processor
def transcribe(model: WhisperForConditionalGeneration, processor: WhisperProcessor, audio: dict) -> str:
    # Process audio to extract input features (returning a PyTorch tensor).
    features = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
    input_features = features.input_features  # Expected shape: [batch_size, num_mel_bins, sequence_length]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_features = input_features.to(device)
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

# Compute word error rate using the evaluate package
def compute_wer(reference: str, hypothesis: str) -> float:
    wer_metric = evaluate.load("wer")
    return wer_metric.compute(predictions=[hypothesis], references=[reference]) * 100

# Evaluate voice lines for overall and phrase-specific error rates.
def evaluate_on_voice_lines(voice_lines: VoiceLines, model: WhisperForConditionalGeneration, processor: WhisperProcessor):
    overall_wer_list = []
    phrase_wer_list = []
    untestable_count = 0

    # Load target phrases from PHRASES_FILE (each line is a phrase or word)
    with open(PHRASES_FILE, "r", encoding="utf-8") as f:
        phrases = [line.strip() for line in f if line.strip()]

    for line in tqdm(voice_lines.lines, desc="Evaluating voice lines", unit="line"):
        audio_path = os.path.join(FILTERED_VOICE_FILE_DIR, line.FileName)
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            continue

        audio = load_audio(audio_path)
        transcription_pred = transcribe(model, processor, audio)
        reference = line.Transcription

        overall_wer = compute_wer(reference, transcription_pred)
        overall_wer_list.append(overall_wer)

        matched_phrases = [phrase for phrase in phrases if voice_line_contains_phrase(line, phrase)]
        if not matched_phrases:
            untestable_count += 1
            continue

        phrase_errors = []
        norm_ref = normalize_text(reference)
        norm_pred = normalize_text(transcription_pred)
        for phrase in matched_phrases:
            norm_phrase = normalize_text(phrase)
            if norm_phrase in norm_ref and norm_phrase in norm_pred:
                phrase_errors.append(0.0)
            else:
                phrase_errors.append(100.0)
        avg_phrase_wer = np.mean(phrase_errors)
        phrase_wer_list.append(avg_phrase_wer)

    avg_overall_wer = np.mean(overall_wer_list) if overall_wer_list else None
    avg_phrase_wer = np.mean(phrase_wer_list) if phrase_wer_list else None

    return avg_overall_wer, avg_phrase_wer, untestable_count, overall_wer_list, phrase_wer_list

# Accumulate transcriptions from both models for each voice line.
def accumulate_transcriptions(voice_lines: VoiceLines,
                              finetuned_model: WhisperForConditionalGeneration, finetuned_processor: WhisperProcessor,
                              baseline_model: WhisperForConditionalGeneration, baseline_processor: WhisperProcessor):
    transcription_results = []
    for line in tqdm(voice_lines.lines, desc="Accumulating transcriptions", unit="line"):
        audio_path = os.path.join(FILTERED_VOICE_FILE_DIR, line.FileName)
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            continue
        audio = load_audio(audio_path)
        tune_transcription = transcribe(finetuned_model, finetuned_processor, audio)
        base_transcription = transcribe(baseline_model, baseline_processor, audio)
        transcription_results.append({
            "base": base_transcription,
            "tune": tune_transcription,
            "gtgt": line.Transcription
        })
    return transcription_results

def run_evaluation(checkpoint_dir: str, test_files: list, baseline_model, baseline_processor):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Evaluating checkpoint: {checkpoint_dir} ===")
    # Load finetuned model from checkpoint.
    finetuned_model = WhisperForConditionalGeneration.from_pretrained(checkpoint_dir).to(device)
    if os.path.exists(os.path.join(checkpoint_dir, "tokenizer_config.json")):
        finetuned_processor = WhisperProcessor.from_pretrained(checkpoint_dir, language="english", task="transcribe")
    else:
        print("Tokenizer files not found in finetuned checkpoint. Falling back to base model tokenizer.")
        finetuned_processor = WhisperProcessor.from_pretrained("openai/whisper-base.en", language="english", task="transcribe")

    # Containers for aggregated results.
    finetuned_overall_wers = []
    finetuned_phrase_wers = []
    baseline_overall_wers = []
    baseline_phrase_wers = []
    finetuned_untestable_total = 0
    baseline_untestable_total = 0
    all_transcription_results = []

    for test_file in test_files:
        print(f"\n--- Processing {test_file} ---")
        voice_lines = VoiceLines.load_from_yaml(test_file)
        print(f"Loaded {len(voice_lines)} voice lines from {test_file}")

        # Evaluate finetuned model.
        (avg_overall_wer_ft, avg_phrase_wer_ft, untestable_ft,
         overall_wer_list_ft, phrase_wer_list_ft) = evaluate_on_voice_lines(voice_lines, finetuned_model, finetuned_processor)
        finetuned_overall_wers.append(avg_overall_wer_ft)
        finetuned_phrase_wers.append(avg_phrase_wer_ft)
        finetuned_untestable_total += untestable_ft

        # Evaluate baseline model.
        (avg_overall_wer_base, avg_phrase_wer_base, untestable_base,
         overall_wer_list_base, phrase_wer_list_base) = evaluate_on_voice_lines(voice_lines, baseline_model, baseline_processor)
        baseline_overall_wers.append(avg_overall_wer_base)
        baseline_phrase_wers.append(avg_phrase_wer_base)
        baseline_untestable_total += untestable_base

        # Accumulate transcription results.
        transcription_results = accumulate_transcriptions(voice_lines, finetuned_model, finetuned_processor, baseline_model, baseline_processor)
        all_transcription_results.extend(transcription_results)

    # Aggregated metrics.
    aggregated_results = {
        "finetuned_overall_wer": float(np.mean(finetuned_overall_wers)) if finetuned_overall_wers else None,
        "finetuned_phrase_wer": float(np.mean(finetuned_phrase_wers)) if finetuned_phrase_wers else None,
        "finetuned_untestable_count": finetuned_untestable_total,
        "baseline_overall_wer": float(np.mean(baseline_overall_wers)) if baseline_overall_wers else None,
        "baseline_phrase_wer": float(np.mean(baseline_phrase_wers)) if baseline_phrase_wers else None,
        "baseline_untestable_count": baseline_untestable_total
    }

    print("\n=== Aggregated Evaluation Results ===")
    print("Finetuned Model:")
    print(f"  Average Overall WER: {aggregated_results['finetuned_overall_wer']:.2f}%")
    print(f"  Average Phrase Error: {aggregated_results['finetuned_phrase_wer']:.2f}%")
    print(f"  Untestable Voice Lines: {aggregated_results['finetuned_untestable_count']}")
    print("\nBaseline Model (whisper-base.en):")
    print(f"  Average Overall WER: {aggregated_results['baseline_overall_wer']:.2f}%")
    print(f"  Average Phrase Error: {aggregated_results['baseline_phrase_wer']:.2f}%")
    print(f"  Untestable Voice Lines: {aggregated_results['baseline_untestable_count']}")

    # Prepare final result dictionary.
    results = {
        "aggregated_results": aggregated_results,
        "transcriptions": all_transcription_results
    }
    return results

def main():
    # List the test YAML file(s)
    test_files = ["2-test.yaml"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load the baseline model and processor once.
    baseline_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en").to(device)
    baseline_processor = WhisperProcessor.from_pretrained("openai/whisper-base.en", language="english", task="transcribe")

    # List of checkpoint directories to evaluate.
    checkpoint_dirs = [
        "./whisper-skyrim-en/checkpoint-1200",
        "./whisper-skyrim-en/checkpoint-600"
    ]

    for ckpt in checkpoint_dirs:
        results = run_evaluation(ckpt, test_files, baseline_model, baseline_processor)
        output_file = f"transcription_results_{os.path.basename(ckpt)}.yaml"
        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(results, f, sort_keys=False, allow_unicode=True)
        print(f"\nResults for checkpoint {ckpt} written to {output_file}")

if __name__ == "__main__":
    main()
