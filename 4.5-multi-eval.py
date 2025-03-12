#!/usr/bin/env python3
"""
4.3-eval-selected-voice-datasets.py

This script evaluates performance on selected voice datasets from Hugging Face:
  - CommonVoice (mozilla-foundation/common_voice_11_0)
  - AMI (edinburghcstr/ami)
  - Earnings22 (distil-whisper/earnings22)
  - Voxpopuli (facebook/voxpopuli)

For each dataset, it selects a specified number of samples, computes the word error rate (WER)
for each model (the base model "openai/whisper-base.en" and our finetuned checkpoints),
and prints the results to the console.
"""

import os
import torch
import evaluate
from tqdm import tqdm
from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Mapping for transcription field names for each dataset alias.
TRANSCRIPTION_FIELDS = {
    "CommonVoice": "sentence",
    "AMI": "transcript",
    "Earnings22": "transcript",
    "Voxpopuli": "transcript"
}

def transcribe(model, processor, audio_dict):
    """
    Transcribes an audio sample using the given model and processor.
    Expects an audio dictionary with keys "array" and "sampling_rate".
    """
    features = processor.feature_extractor(
        audio_dict["array"],
        sampling_rate=audio_dict["sampling_rate"],
        return_tensors="pt"
    )
    input_features = features.input_features
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_features = input_features.to(device)
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def evaluate_dataset(model, processor, dataset, dataset_alias, num_samples=15000):
    """
    Evaluates the given model on the first num_samples of the provided dataset.
    The transcription reference field is chosen based on the dataset alias.
    Returns the computed WER (in percentage) on these samples.
    """
    wer_metric = evaluate.load("wer")
    total_references = []
    total_predictions = []
    ref_field = TRANSCRIPTION_FIELDS[dataset_alias]
    
    for sample in tqdm(dataset, total=num_samples, desc=f"Evaluating {dataset_alias}", unit="sample"):
        audio = sample["audio"]
        reference = sample[ref_field]
        prediction = transcribe(model, processor, audio)
        total_references.append(reference)
        total_predictions.append(prediction)
        if len(total_references) >= num_samples:
            break
    wer = wer_metric.compute(predictions=total_predictions, references=total_references) * 100
    return wer

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_samples = 1  # Number of samples per dataset
    
    # Information for each dataset.
    # Note: The CommonVoice entry is added below.
    datasets_info = {
        "CommonVoice": {"dataset_id": "mozilla-foundation/common_voice_11_0", "config": "en", "split": "test", "streaming": True},
        "Earnings22": {"dataset_id": "distil-whisper/earnings22", "config": None, "split": "test", "streaming": True},
        "Voxpopuli": {"dataset_id": "facebook/voxpopuli", "config": "en", "split": "test", "streaming": True}
    }
    
    # Define models to evaluate.
    checkpoint_dirs = [
        "./whisper-skyrim-en/8-3FreezeDec2",
    ]
    models_to_evaluate = {"base": "openai/whisper-base.en"}
    for ckpt in checkpoint_dirs:
        models_to_evaluate[os.path.basename(ckpt)] = ckpt
    
    results = {}  # Will store results as: results[model_name][dataset_alias] = wer
    
    for model_name, ckpt in models_to_evaluate.items():
        print(f"\nEvaluating model: {model_name}")
        model = WhisperForConditionalGeneration.from_pretrained(ckpt).to(device)
        if os.path.exists(os.path.join(ckpt, "tokenizer_config.json")):
            processor = WhisperProcessor.from_pretrained(ckpt, language="english", task="transcribe")
        else:
            processor = WhisperProcessor.from_pretrained("openai/whisper-base.en", language="english", task="transcribe")
        
        results[model_name] = {}
        for ds_alias, ds_info in datasets_info.items():
            print(f"\nLoading {ds_alias} test set...")
            if ds_info.get("streaming"):
                dataset_obj = load_dataset(
                    ds_info["dataset_id"], 
                    ds_info["config"], 
                    split=ds_info["split"], 
                    streaming=True,
                    trust_remote_code=True
                )
            else:
                if ds_info["config"]:
                    dataset_obj = load_dataset(
                        ds_info["dataset_id"], 
                        ds_info["config"], 
                        split=ds_info["split"],
                        trust_remote_code=True
                    )
                else:
                    dataset_obj = load_dataset(
                        ds_info["dataset_id"], 
                        split=ds_info["split"],
                        trust_remote_code=True
                    )
            
            dataset_obj = dataset_obj.cast_column("audio", Audio(sampling_rate=16000))
            wer = evaluate_dataset(model, processor, dataset_obj, ds_alias, num_samples=num_samples)
            results[model_name][ds_alias] = wer
            print(f"{ds_alias} WER: {wer:.2f}%")
    
    print("\n=== Evaluation Results ===")
    for model_name, ds_results in results.items():
        print(f"\nModel: {model_name}")
        for ds_alias, wer in ds_results.items():
            print(f"{ds_alias}: {wer:.2f}%")
    
if __name__ == "__main__":
    main()
