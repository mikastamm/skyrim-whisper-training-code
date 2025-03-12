#!/usr/bin/env python3
import os
import sys
import yaml
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, List, Optional
from datasets import Dataset, DatasetDict, Audio, load_dataset, concatenate_datasets
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import evaluate
from huggingface_hub import login
from audiomentations import (
    AddGaussianNoise,
    Compose,
    Gain,
    OneOf,
    PitchShift,
    TimeStretch,
)
from Pipeline import VoiceLines, VoiceLine, print_stage_header, FILTERED_VOICE_FILE_DIR
import time
import multiprocessing

# Configuration
WHISPER_MODEL = "openai/whisper-base.en"  # Can be changed to other Whisper models
LANGUAGE = "english"  # Language for transcription
TASK = "transcribe"  # Task type
OUTPUT_DIR = "./whisper-skyrim-en"  # Output directory for model checkpoints
TRAINING_STEPS = 3000  # Total training steps
EVAL_STEPS = 600  # Evaluate every N steps
SAVE_STEPS = 600  # Save checkpoint every N steps
TRAIN_BATCH_SIZE = 24  # Batch size for training
EVAL_BATCH_SIZE = 12  # Batch size for evaluation
PUSH_TO_HUB = True  # Whether to push models to Hugging Face Hub
HF_TOKEN = None  # Set to your Hugging Face token or None to prompt
REPO_NAME = "whisper-skyrim-en"  # Repository name on Hugging Face Hub
TRAIN_YAML = "2-train.yaml"  # Path to training data
VALIDATION_YAML = "2-validation.yaml"  # Path to validation data
TEST_YAML = "2-test.yaml"  # Path to test data
AUDIO_COLUMN_NAME = "audio"  # Column name for audio data
TEXT_COLUMN_NAME = "sentence"  # Column name for transcription text
NUM_PROC = 1  # Number of processes for dataset operations (set to 1 to avoid multiprocessing issues)

# Configurable percentage (between 0 and 1) of common-voice-en examples to add
commonVoicePct01 = 0.0

def main():
    print_stage_header("Stage 2: Training Whisper on Skyrim Dataset")

    # Try to log in to Hugging Face Hub if pushing to hub
    global PUSH_TO_HUB
    if PUSH_TO_HUB:
        try:
            if HF_TOKEN:
                login(token=HF_TOKEN)
            else:
                token = input("\nEnter your Hugging Face token (or press Enter to skip Hub upload): ")
                if token.strip():
                    login(token=token)
                else:
                    print("No token provided. Models won't be pushed to the Hub.")
                    PUSH_TO_HUB = False

            if PUSH_TO_HUB:
                print("Successfully logged in to Hugging Face Hub")
        except Exception as e:
            print(f"Warning: Could not log in to Hugging Face Hub: {e}")
            print("Training will continue but models won't be pushed.")
            PUSH_TO_HUB = False

    # Load the model to get configuration
    print(f"Loading Whisper model ({WHISPER_MODEL}) to get configuration...")
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    tokenizer = WhisperTokenizer.from_pretrained(WHISPER_MODEL, language=LANGUAGE, task=TASK)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to end-of-sentence token
    max_label_length = model.config.max_length
    print(f"Maximum token length supported by model: {max_label_length}")

    # Freeze encoder parameters so they are not updated during training
    print("Freezing encoder parameters...")
    encoder = model.get_encoder() if hasattr(model, "get_encoder") else model.model.encoder
    for param in encoder.parameters():
        param.requires_grad = False
    
    decoder = model.get_decoder() if hasattr(model, "get_decoder") else model.model.decoder
    freeze_decoder_layers=3
    i=0
       
    for layer in decoder.layers:
        
        for param in layer.parameters():
            param.requires_grad = False
            
        i+=1
        if i==freeze_decoder_layers:
            break

    print(f"Freezed {i} layers of decoder")

    # Load the Skyrim data from YAML files and prepare dataset
    def load_skyrim_data(split: str) -> Dataset:
        """
        Load data from YAML files and create a dataset with audio and transcription.
        Format: ['audio', 'sentence', 'plugin', 'voice_type', 'duration_ms', 'internal_file_name']
        """
        file_path = {
            "train": TRAIN_YAML,
            "validation": VALIDATION_YAML,
            "test": TEST_YAML
        }.get(split)

        if not file_path or not os.path.exists(file_path):
            raise ValueError(f"Invalid split '{split}' or file not found")

        # Load YAML data
        voice_lines = VoiceLines.load_from_yaml(file_path)

        # Create lists for features
        audio_paths = []
        sentences = []
        plugins = []        # Store plugin info for augmentation
        voice_types = []    # Store voice type for metrics
        durations_ms = []   # Store duration for metrics
        internal_file_names = []  # Store internal filename for identification

        print(f"Checking {len(voice_lines.lines)} voice lines for {split} split...")

        # Track voice lines that exceed token length
        long_sentences = 0
        missing_files = 0

        # Extract data from VoiceLines
        for line in voice_lines.lines:
            # Ensure the voice line is less than 30s long
            if line.DurationMs >= 30000:
                continue

            # Pre-check if the sentence would exceed token length
            tokens = tokenizer(line.Transcription).input_ids
            if len(tokens) >= max_label_length:
                long_sentences += 1
                continue

            audio_path = os.path.join(FILTERED_VOICE_FILE_DIR, line.FileName)
            if os.path.exists(audio_path):
                audio_paths.append(audio_path)
                sentences.append(line.Transcription)
                plugins.append(line.Plugin)
                voice_types.append(line.VoiceType)
                durations_ms.append(line.DurationMs)
                internal_file_names.append(line.InternalFileName)
            else:
                missing_files += 1

        if long_sentences > 0:
            print(f"Removed {long_sentences} voice lines that exceed the maximum token length ({max_label_length})")
        if missing_files > 0:
            print(f"Warning: {missing_files} audio files were not found")

        # Create a dataset
        data = {
            "audio": audio_paths,
            TEXT_COLUMN_NAME: sentences,
            "plugin": plugins,
            "voice_type": voice_types,
            "duration_ms": durations_ms,
            "internal_file_name": internal_file_names
        }

        dataset = Dataset.from_dict(data)
        # Add audio loading capability
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        print(f"Loaded {len(dataset)} examples for {split} split")
        return dataset

    # Load datasets
    print("Loading datasets...")
    dataset = DatasetDict()
    dataset["train"] = load_skyrim_data("train")
    dataset["validation"] = load_skyrim_data("validation")
    dataset["test"] = load_skyrim_data("test")

    # --- New: Sample and add common-voice-en data using streaming ---
    def sample_common_voice_data(split: str, target_count: int):
        """
        Sample a target number of examples from common-voice-en using streaming.
        Compute duration on the fly and bring the examples into the format:
        ['audio', 'sentence', 'plugin', 'voice_type', 'duration_ms', 'internal_file_name'].
        Only include examples with computed duration < 30s and prioritize diversity (one per speaker).
        """
        print(f"Sampling {target_count} examples from common-voice-en ({split} split)...")
        ds_stream = load_dataset("mozilla-foundation/common_voice_11_0", "en", split=split, streaming=True)
        samples = []
        seen_speakers = set()
        for example in ds_stream:
            audio_data = example["audio"]
            if "array" not in audio_data or "sampling_rate" not in audio_data:
                continue
            duration_sec = len(audio_data["array"]) / audio_data["sampling_rate"]
            if duration_sec >= 30:
                continue
            speaker = example["client_id"]
            if speaker in seen_speakers:
                continue
            seen_speakers.add(speaker)
            samples.append({
                "audio": audio_data,
                "sentence": example["sentence"],
                "plugin": "common_voice",
                "voice_type": speaker,
                "duration_ms": int(duration_sec * 1000),
                "internal_file_name": f"common_voice_{speaker}"
            })
            if len(samples) >= target_count:
                break
        return Dataset.from_list(samples)

    if commonVoicePct01 > 0:
        # For training set
        num_skyrim_train = len(dataset["train"])
        target_common_train = int(num_skyrim_train * commonVoicePct01)
        common_voice_train = sample_common_voice_data("train", target_common_train)
        common_voice_train = common_voice_train.cast_column("audio", Audio(sampling_rate=16000))
        dataset["train"] = concatenate_datasets([dataset["train"], common_voice_train])
        print(f"Added {len(common_voice_train)} common-voice-en examples to the training set.")

        # For validation set
        num_skyrim_val = len(dataset["validation"])
        target_common_val = int(num_skyrim_val * commonVoicePct01)
        common_voice_val = sample_common_voice_data("validation", target_common_val)
        common_voice_val = common_voice_val.cast_column("audio", Audio(sampling_rate=16000))
        dataset["validation"] = concatenate_datasets([dataset["validation"], common_voice_val])
        print(f"Added {len(common_voice_val)} common-voice-en examples to the validation set.")

    # Downstream processing treats all data the same.
    # Function to prepare dataset for training
    def prepare_dataset(batch):
        """Prepare a batch for training by computing features and encoding text"""
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = tokenizer(batch[TEXT_COLUMN_NAME]).input_ids
        return batch

    # Load the pre-trained model components
    print(f"Loading Whisper model components ({WHISPER_MODEL})...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL)
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL, language=LANGUAGE, task=TASK)

    # Process the datasets (all examples are now in a uniform format)
    print("Processing datasets...")
    columns_to_remove = [col for col in dataset["train"].column_names if col not in ["labels", "input_features"]]
    processed_dataset = dataset.map(prepare_dataset, remove_columns=columns_to_remove, num_proc=NUM_PROC)

    # Create data collator
    class DataCollatorSpeechSeq2SeqWithPadding:
        def __init__(self, processor, decoder_start_token_id):
            self.processor = processor
            self.decoder_start_token_id = decoder_start_token_id

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Load the WER metric for evaluation
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        num_train_epochs=8,
        max_steps=TRAINING_STEPS,
        warmup_steps=1500,
        lr_scheduler_type="linear",
        weight_decay=0.0075,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        logging_steps=50,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=PUSH_TO_HUB,
        hub_model_id=REPO_NAME,
        dataloader_num_workers=4
    )

    # Initialize the Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    # Start training
    print("Starting training...")
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time

    print(f"Training completed in {training_time/60:.1f} minutes")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))

    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(processed_dataset["test"])
    print(f"Test WER: {test_results['eval_wer']:.2f}%")
    print("Training complete!")

if __name__ == "__main__":
    if sys.platform == 'win32':
        multiprocessing.freeze_support()
    main()
