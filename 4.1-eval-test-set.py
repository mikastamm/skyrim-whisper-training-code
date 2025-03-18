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

# Load audio using soundfile.
def load_audio(file_path: str) -> dict:
    audio_array, sr = sf.read(file_path)
    return {"array": audio_array, "sampling_rate": sr}

# Transcribe an audio sample using the given model and processor.
def transcribe(model: WhisperForConditionalGeneration, processor: WhisperProcessor, audio: dict) -> str:
    features = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
    input_features = features.input_features  # Shape: [batch_size, num_mel_bins, sequence_length]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_features = input_features.to(device)
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def evaluate_and_accumulate_voice_lines(voice_lines: VoiceLines,
                                        finetuned_model: WhisperForConditionalGeneration, finetuned_processor: WhisperProcessor,
                                        baseline_model: WhisperForConditionalGeneration, baseline_processor: WhisperProcessor,
                                        precomputed_phrases: list, wer_metric) -> tuple:
    """
    For each voice line, load the audio once and use it for both models.
    Compute overall WER and phrase error (if applicable) using precomputed phrases.
    Also, accumulate transcription results.
    """
    overall_wer_list_ft = []
    phrase_wer_list_ft = []
    untestable_ft = 0

    overall_wer_list_base = []
    phrase_wer_list_base = []
    untestable_base = 0

    transcription_results = []

    for line in tqdm(voice_lines.lines, desc="Processing voice lines", unit="line"):
        audio_path = os.path.join(FILTERED_VOICE_FILE_DIR, line.FileName)
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            continue
        audio = load_audio(audio_path)
        reference = line.Transcription
        norm_ref = normalize_text(reference)

        # Determine which phrases are present in the reference.
        matched_phrases = []
        for entry in precomputed_phrases:
            if entry["pattern"] is not None:
                if entry["pattern"].search(norm_ref) is not None:
                    matched_phrases.append(entry)
            else:
                if entry["norm_phrase"] in norm_ref:
                    matched_phrases.append(entry)

        # Finetuned model processing.
        tune_transcription = transcribe(finetuned_model, finetuned_processor, audio)
        overall_wer_ft = wer_metric.compute(predictions=[tune_transcription], references=[reference]) * 100
        overall_wer_list_ft.append(overall_wer_ft)
        if matched_phrases:
            norm_tune = normalize_text(tune_transcription)
            # For each phrase present in the reference, error is 0 if found in prediction, else 100.
            phrase_errors = [0.0 if entry["norm_phrase"] in norm_tune else 100.0 for entry in matched_phrases]
            phrase_wer_list_ft.append(np.mean(phrase_errors))
        else:
            untestable_ft += 1

        # Baseline model processing.
        base_transcription = transcribe(baseline_model, baseline_processor, audio)
        overall_wer_base = wer_metric.compute(predictions=[base_transcription], references=[reference]) * 100
        overall_wer_list_base.append(overall_wer_base)
        if matched_phrases:
            norm_base = normalize_text(base_transcription)
            phrase_errors = [0.0 if entry["norm_phrase"] in norm_base else 100.0 for entry in matched_phrases]
            phrase_wer_list_base.append(np.mean(phrase_errors))
        else:
            untestable_base += 1

        # Accumulate both transcriptions.
        transcription_results.append({
            "base": base_transcription,
            "tune": tune_transcription,
            "gtgt": reference
        })

    avg_overall_wer_ft = np.mean(overall_wer_list_ft) if overall_wer_list_ft else None
    avg_phrase_wer_ft = np.mean(phrase_wer_list_ft) if phrase_wer_list_ft else None
    avg_overall_wer_base = np.mean(overall_wer_list_base) if overall_wer_list_base else None
    avg_phrase_wer_base = np.mean(phrase_wer_list_base) if phrase_wer_list_base else None

    return (avg_overall_wer_ft, avg_phrase_wer_ft, untestable_ft, overall_wer_list_ft, phrase_wer_list_ft,
            avg_overall_wer_base, avg_phrase_wer_base, untestable_base, overall_wer_list_base, phrase_wer_list_base,
            transcription_results)

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

    # Global accumulators for all test files.
    all_overall_wer_list_ft = []
    all_phrase_wer_list_ft = []
    all_untestable_ft = 0

    all_overall_wer_list_base = []
    all_phrase_wer_list_base = []
    all_untestable_base = 0

    all_transcription_results = []

    # Load the WER metric only once.
    wer_metric = evaluate.load("wer")

    for test_file in test_files:
        print(f"\n--- Processing {test_file} ---")
        voice_lines = VoiceLines.load_from_yaml(test_file)
        print(f"Loaded {len(voice_lines)} voice lines from {test_file}")

        # Load target phrases once and precompute normalized forms and regex patterns.
        with open(PHRASES_FILE, "r", encoding="utf-8") as f:
            phrases = [line.strip() for line in f if line.strip()]
        precomputed_phrases = []
        for phrase in phrases:
            norm_phrase = normalize_text(phrase)
            pattern = (re.compile(r'(?:^|[ \?\.\!,])' + re.escape(norm_phrase) + r'(?=$|[ \?\.\!,])')
                       if len(norm_phrase) < 5 else None)
            precomputed_phrases.append({"phrase": phrase, "norm_phrase": norm_phrase, "pattern": pattern})

        (avg_overall_wer_ft, avg_phrase_wer_ft, untestable_ft, overall_wer_list_ft, phrase_wer_list_ft,
         avg_overall_wer_base, avg_phrase_wer_base, untestable_base, overall_wer_list_base, phrase_wer_list_base,
         transcription_results) = evaluate_and_accumulate_voice_lines(
            voice_lines, finetuned_model, finetuned_processor,
            baseline_model, baseline_processor,
            precomputed_phrases, wer_metric
         )

        all_overall_wer_list_ft.extend(overall_wer_list_ft)
        all_phrase_wer_list_ft.extend(phrase_wer_list_ft)
        all_untestable_ft += untestable_ft

        all_overall_wer_list_base.extend(overall_wer_list_base)
        all_phrase_wer_list_base.extend(phrase_wer_list_base)
        all_untestable_base += untestable_base

        all_transcription_results.extend(transcription_results)

    aggregated_results = {
        "finetuned_overall_wer": float(np.mean(all_overall_wer_list_ft)) if all_overall_wer_list_ft else None,
        "finetuned_phrase_wer": float(np.mean(all_phrase_wer_list_ft)) if all_phrase_wer_list_ft else None,
        "finetuned_untestable_count": all_untestable_ft,
        "baseline_overall_wer": float(np.mean(all_overall_wer_list_base)) if all_overall_wer_list_base else None,
        "baseline_phrase_wer": float(np.mean(all_phrase_wer_list_base)) if all_phrase_wer_list_base else None,
        "baseline_untestable_count": all_untestable_base
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
        "./whisper-skyrim-en/8-3FreezeDec2",
    ]

    for ckpt in checkpoint_dirs:
        results = run_evaluation(ckpt, test_files, baseline_model, baseline_processor)
        output_file = f"transcription_results_{os.path.basename(ckpt)}.yaml"
        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(results, f, sort_keys=False, allow_unicode=True)
        print(f"\nResults for checkpoint {ckpt} written to {output_file}")

if __name__ == "__main__":
    main()
