#!/usr/bin/env python3
"""
4.3-eval-common-voice-en.py

This script evaluates whether our performance on normal speech data (Common Voice English) has dropped.
It downloads the Common Voice English test split via Hugging Face datasets, selects 2000 samples,
and computes the word error rate (WER) for each model: the base model ("openai/whisper-base.en")
and our finetuned models from specified checkpoint directories. The results are printed to the console.
Now it also supports evaluating faster-whisper checkpoints.
"""

import os
import torch
import numpy as np
import evaluate
import tempfile
import soundfile as sf
from tqdm import tqdm
from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

def transcribe(model: WhisperForConditionalGeneration, processor: WhisperProcessor, audio_dict: dict) -> str:
    """
    Transcribes an audio sample using the given model and processor.
    Expects an audio dictionary with keys "array" and "sampling_rate".
    """
    features = processor.feature_extractor(audio_dict["array"], sampling_rate=audio_dict["sampling_rate"], return_tensors="pt")
    input_features = features.input_features  # Expected shape: [batch_size, num_mel_bins, sequence_length]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_features = input_features.to(device)
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def transcribe_faster(model, audio_dict: dict) -> str:
    """
    Transcribes an audio sample using the faster-whisper model.
    Since faster-whisper expects a file path, this function writes the audio to a temporary WAV file.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio_dict["array"], audio_dict["sampling_rate"])
        temp_path = tmp.name
    segments, _ = model.transcribe(temp_path, beam_size=5)
    transcription = " ".join([seg.text for seg in segments])
    os.remove(temp_path)
    return transcription

def evaluate_common_voice(model, processor, dataset, num_samples=15000, use_faster_whisper: bool = False) -> float:
    """
    Evaluates the given model on the first num_samples of the Common Voice English test set.
    If use_faster_whisper is True, the faster-whisper transcribe function is used.
    Returns the computed WER (in percentage) on these samples.
    """
    wer_metric = evaluate.load("wer")
    total_references = []
    total_predictions = []
    
    for sample in tqdm(dataset, total=num_samples, desc="Evaluating Common Voice", unit="sample"):
        # In Common Voice, the transcription is typically in the "sentence" field.
        audio = sample["audio"]
        reference = sample["sentence"]
        if use_faster_whisper:
            prediction = transcribe_faster(model, audio)
        else:
            prediction = transcribe(model, processor, audio)
        total_references.append(reference)
        total_predictions.append(prediction)
        if len(total_references) >= num_samples:
            break
    wer = wer_metric.compute(predictions=total_predictions, references=total_references) * 100
    return wer

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading Common Voice English test set...")
    # Load the Common Voice English test split (using version 11.0)
    dataset = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="test", streaming=True)
    # Ensure the audio column is decoded with a 16 kHz sampling rate.
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    num_samples = 15000
    results = {}
    
    # Optionally, evaluate the base model (commented out here).
    # print("\nEvaluating base model (openai/whisper-base.en)...")
    # base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en").to(device)
    # base_processor = WhisperProcessor.from_pretrained("openai/whisper-base.en", language="english", task="transcribe")
    # wer_base = evaluate_common_voice(base_model, base_processor, dataset, num_samples=num_samples)
    # results["base"] = wer_base
    # print(f"Base model WER: {wer_base:.2f}%")
    
    # Configurable list of normal whisper checkpoint directories.
    checkpoint_dirs = [
        "sorendal/skyrim-whisper-small",
        "openai/whisper-small",
        # "./whisper-skyrim-en/checkpoint-1200",
    ]
        # Configurable list of faster-whisper checkpoint directories.
    faster_whisper_checkpoint_dirs = [
        "sorendal/skyrim-whisper-small-int8",
    ]
    # Evaluate each normal whisper checkpoint.
    for ckpt in checkpoint_dirs:
        print(f"\nEvaluating finetuned model from checkpoint: {ckpt}")
        model = WhisperForConditionalGeneration.from_pretrained(ckpt).to(device)
        if os.path.exists(os.path.join(ckpt, "tokenizer_config.json")):
            processor = WhisperProcessor.from_pretrained("openai/whisper-base.en", language="english", task="transcribe")
        else:
            print("Tokenizer files not found in checkpoint; falling back to base model processor.")
            processor = WhisperProcessor.from_pretrained("openai/whisper-base.en", language="english", task="transcribe")
        wer_ckpt = evaluate_common_voice(model, processor, dataset, num_samples=num_samples, use_faster_whisper=False)
        results[os.path.basename(ckpt)] = wer_ckpt
        print(f"Checkpoint {os.path.basename(ckpt)} WER: {wer_ckpt:.2f}%")
    

    
    # Evaluate each faster-whisper checkpoint.
    for ckpt in faster_whisper_checkpoint_dirs:
        print(f"\nEvaluating faster-whisper model from checkpoint: {ckpt}")
        from faster_whisper import WhisperModel
        model = WhisperModel(ckpt, device=device)
        # Processor is not used for faster-whisper.
        wer_ckpt = evaluate_common_voice(model, None, dataset, num_samples=num_samples, use_faster_whisper=True)
        results["faster_" + os.path.basename(ckpt)] = wer_ckpt
        print(f"Faster checkpoint {os.path.basename(ckpt)} WER: {wer_ckpt:.2f}%")
    
    print("\n=== Evaluation Results on Common Voice English ===")
    for key, value in results.items():
        print(f"{key}: {value:.2f}%")

if __name__ == "__main__":
    main()
