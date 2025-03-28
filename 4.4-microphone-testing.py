#!/usr/bin/env python3
"""
4.4-microphone-testing.py

This script waits for you to press ENTER to start recording audio from the microphone,
and then again to stop recording. Once an utterance is captured, it transcribes the audio 
using both the base Whisper model and one or more finetuned models (specified via a configuration variable at the top).
It then compares the transcriptions by detecting target phrases (loaded from PHRASES_FILE), 
highlights these phrases in color, and if a phrase is detected by one model but not by another, it marks that as an error.
"""

import os
import re
import queue
import numpy as np
import sounddevice as sd
import torch
from termcolor import colored
import colorama
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from Pipeline import PHRASES_FILE  # Assumes PHRASES_FILE is defined in your Pipeline module

# Initialize colorama for Windows.
colorama.init()

# ==================== Configuration ====================
CHECKPOINT_DIRS = [
    "./whisper-skyrim-en/checkpoint-1200",
    "./whisper-skyrim-en/checkpoint-600"
]
SAMPLE_RATE = 16000  # 16 kHz
CHANNELS = 1
# =======================================================

# Load target phrases.
with open(PHRASES_FILE, "r", encoding="utf-8") as f:
    TARGET_PHRASES = [line.strip() for line in f if line.strip()]

def normalize_text(text: str) -> str:
    """Lowercase, replace hyphens with spaces, and remove non-alphanumeric characters."""
    text = text.lower().replace('-', ' ')
    return re.sub(r'[^0-9a-z\s]', '', text)

def highlight_phrases(text: str, phrases: list, color="yellow"):
    """
    Return text with each occurrence (case-insensitive) of a phrase replaced with a colored version.
    """
    highlighted = text
    for phrase in phrases:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        highlighted = pattern.sub(lambda m: colored(m.group(0), color, attrs=["bold"]), highlighted)
    return highlighted

def detect_phrases(text: str, phrases: list) -> set:
    """
    Return a set of normalized phrases from the given list that appear in text.
    """
    norm_text = normalize_text(text)
    detected = set()
    for phrase in phrases:
        if normalize_text(phrase) in norm_text:
            detected.add(normalize_text(phrase))
    return detected

def record_utterance():
    """
    Records audio from the microphone when the user presses ENTER to start,
    and stops recording when the user presses ENTER again.
    Returns a dict with keys "array" (numpy array of samples) and "sampling_rate".
    """
    input(colored("Press ENTER to start recording...", "green", attrs=["bold"]))
    print(colored("Recording... Press ENTER to stop recording.", "green", attrs=["bold"]))
    
    q = queue.Queue()
    utterance_blocks = []

    def callback(indata, frames, time_info, status):
        q.put(indata.copy())

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32", callback=callback)
    stream.start()

    # Wait for user to press ENTER to stop recording.
    input()

    stream.stop()
    stream.close()

    # Drain the audio queue.
    while not q.empty():
        block = q.get()
        utterance_blocks.append(block.flatten())
    
    if utterance_blocks:
        utterance_np = np.concatenate(utterance_blocks, axis=0)
    else:
        utterance_np = np.array([])

    return {"array": utterance_np, "sampling_rate": SAMPLE_RATE}

def transcribe_audio(model, processor, audio):
    """
    Transcribes the given audio dict using the provided model and processor.
    """
    features = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
    input_features = features.input_features  # shape: [batch, num_mel_bins, sequence_length]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_features = input_features.to(device)
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def load_models():
    """
    Loads the base Whisper model and processor as well as a list of finetuned models (and their processors)
    from the configured CHECKPOINT_DIRS.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading base model...")
    base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en").to(device)
    base_processor = WhisperProcessor.from_pretrained("openai/whisper-base.en", language="english", task="transcribe")
    
    finetuned_models = []  # List of tuples: (name, model, processor)
    for ckpt_dir in CHECKPOINT_DIRS:
        print(f"Loading finetuned model from {ckpt_dir}...")
        model = WhisperForConditionalGeneration.from_pretrained(ckpt_dir).to(device)
        if os.path.exists(os.path.join(ckpt_dir, "tokenizer_config.json")):
            processor = WhisperProcessor.from_pretrained(ckpt_dir, language="english", task="transcribe")
        else:
            print(f"Tokenizer files not found in {ckpt_dir}; falling back to base processor.")
            processor = base_processor
        model_name = os.path.basename(ckpt_dir)
        finetuned_models.append((model_name, model, processor))
    
    return base_model, base_processor, finetuned_models

def main():
    base_model, base_processor, finetuned_models = load_models()
    
    while True:
        print("\n" + colored("=== Ready for a new recording ===", "magenta", attrs=["bold"]))
        print("Press Ctrl+C to exit.")
        audio = record_utterance()
        
        print(colored("Transcribing with base model...", "blue"))
        base_transcription = transcribe_audio(base_model, base_processor, audio)
        
        finetuned_results = {}
        for model_name, model, processor in finetuned_models:
            print(colored(f"Transcribing with finetuned model {model_name}...", "blue"))
            transcription = transcribe_audio(model, processor, audio)
            finetuned_results[model_name] = transcription
        
        base_detected = detect_phrases(base_transcription, TARGET_PHRASES)
        finetuned_detected = {}
        for model_name, transcription in finetuned_results.items():
            finetuned_detected[model_name] = detect_phrases(transcription, TARGET_PHRASES)
        
        # Build union of detected phrases.
        union_phrases = set(base_detected)
        for detected in finetuned_detected.values():
            union_phrases = union_phrases.union(detected)
        
        # Compute errors: phrases missing relative to the union.
        errors = {}
        errors["base"] = union_phrases - base_detected
        for model_name, detected in finetuned_detected.items():
            errors[model_name] = union_phrases - detected
        
        # Highlight target phrases in transcriptions.
        base_highlighted = highlight_phrases(base_transcription, TARGET_PHRASES, color="yellow")
        finetuned_highlighted = {name: highlight_phrases(trans, TARGET_PHRASES, color="yellow") 
                                 for name, trans in finetuned_results.items()}
        
        # Output results.
        print("\n" + colored("=== Transcription Results ===", "green", attrs=["bold"]))
        print(colored("Base Model:", "cyan", attrs=["bold"]))
        print(base_highlighted)
        for model_name, text in finetuned_highlighted.items():
            print(colored(f"Finetuned Model ({model_name}):", "cyan", attrs=["bold"]))
            print(text)
        
        print("\n" + colored("Detected Phrases:", "green", attrs=["bold"]))
        print("Union of detected phrases:", list(union_phrases))
        print("Base model detected:", list(base_detected))
        for model_name, detected in finetuned_detected.items():
            print(f"Finetuned model {model_name} detected:", list(detected))
        
        for model_key, missing in errors.items():
            if missing:
                print(colored(f"ERROR: {model_key} is missing phrases:", "red"), list(missing))
        
        try:
            input(colored("\nPress Enter to record another utterance...", "magenta"))
        except KeyboardInterrupt:
            print("\nExiting.")
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting microphone testing.")
