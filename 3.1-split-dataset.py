#!/usr/bin/env python3
import sys
import os
import random
import re
import yaml
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple
from Pipeline import VoiceLines, VoiceLine, print_stage_header, PHRASES_FILE

# Configuration
VALIDATION_VOICES = []  # Voices exclusive to validation set
TEST_VOICES = ["3dnpcvoice"]  # Voices exclusive to test set
TRAIN_RATIO01 = 0.7
VALIDATION_RATIO01 = 0.2
TEST_RATIO01 = 0.1
HIGH_DENSITY_TEST_LINES = 5  # Number of high-density lines to add to test set
INPUT_FILE = "2-training-data.yaml"
TRAIN_OUTPUT = "2-train.yaml"
VALIDATION_OUTPUT = "2-validation.yaml"
TEST_OUTPUT = "2-test.yaml"
RANDOM_SEED = 43  # For reproducibility

# Set random seed for reproducibility
random.seed(RANDOM_SEED)

print_stage_header("Stage 2: Splitting Dataset for Training")

# ===== Helper Functions for Normalization and Matching =====
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
        # Use a non-capturing group to match start-of-string or allowed punctuation before the phrase
        pattern = r'(?:^|[ \?\.\!,])' + re.escape(norm_phrase) + r'(?=$|[ \?\.\!,])'
        return re.search(pattern, norm_transcription) is not None
    else:
        return norm_phrase in norm_transcription

def count_phrases_in_voice_line(line: VoiceLine, phrases: List[str]) -> int:
    """
    Count how many unique phrases from the list appear in the voice line.
    """
    return sum(1 for phrase in phrases if voice_line_contains_phrase(line, phrase))

def get_phrase_density(line: VoiceLine, phrases: List[str]) -> float:
    """
    Calculate phrase density: number of phrases divided by transcription length.
    """
    phrase_count = count_phrases_in_voice_line(line, phrases)
    if not line.Transcription:
        return 0
    # Use normalized text length for consistent comparison
    norm_transcription = normalize_text(line.Transcription)
    length = len(norm_transcription.split())
    return phrase_count / max(1, length)  # Avoid division by zero

# Load filtered voice lines from the previous stage
voice_lines = VoiceLines.load_from_yaml(INPUT_FILE)

# Load target phrases from file
if os.path.exists(PHRASES_FILE):
    with open(PHRASES_FILE, 'r', encoding='utf-8') as file:
        phrases = [line.strip() for line in file if line.strip()]
    print(f"Loaded {len(phrases)} target phrases from {PHRASES_FILE}")
else:
    print(f"Warning: Phrases file {PHRASES_FILE} not found. Using empty list.")
    phrases = []

# Organize voice lines by speaker
voice_by_speaker: Dict[str, List[VoiceLine]] = defaultdict(list)
for line in voice_lines.lines:
    voice_by_speaker[line.VoiceType].append(line)

# Randomize the order of voice lines within each speaker
for speaker, lines in voice_by_speaker.items():
    random.shuffle(lines)
    print(f"Randomized order of {len(lines)} voice lines for speaker: {speaker}")

# Initialize sets for each split
train_lines = []
validation_lines = []
test_lines = []

# Track which speakers have been assigned
assigned_speakers = set()

# First, handle exclusive speakers for validation and test
for voice_type in VALIDATION_VOICES:
    if voice_type in voice_by_speaker:
        validation_lines.extend(voice_by_speaker[voice_type])
        assigned_speakers.add(voice_type)
        print(f"Added {len(voice_by_speaker[voice_type])} lines from {voice_type} to validation set (exclusive)")

# Always include test speakers in the test set, regardless of the test ratio
for voice_type in TEST_VOICES:
    if voice_type in voice_by_speaker:
        test_lines.extend(voice_by_speaker[voice_type])
        assigned_speakers.add(voice_type)
        print(f"Added {len(voice_by_speaker[voice_type])} lines from {voice_type} to test set (exclusive)")

# Get speaker's primary plugin
speaker_primary_plugin = {}
for speaker, lines in voice_by_speaker.items():
    plugin_counts = Counter(line.Plugin for line in lines)
    speaker_primary_plugin[speaker] = plugin_counts.most_common(1)[0][0]

# Get all unique plugins
all_plugins = set(line.Plugin for line in voice_lines.lines)

# Get remaining speakers
remaining_speakers = [s for s in voice_by_speaker.keys() if s not in assigned_speakers]

# Calculate how many lines to assign to each split to reach target ratio
total_remaining_lines = sum(len(voice_by_speaker[s]) for s in remaining_speakers)
target_train_lines = int(total_remaining_lines * TRAIN_RATIO01)
target_validation_lines = int(total_remaining_lines * VALIDATION_RATIO01)
target_test_lines = 0 if TEST_RATIO01 == 0 and len(test_lines) > 0 else int(total_remaining_lines * TEST_RATIO01)

print(f"Target distribution: Train={target_train_lines}, Validation={target_validation_lines}, Test={target_test_lines}")

# Function to calculate current plugin distribution
def get_plugin_distribution(split_lines):
    if not split_lines:
        return {plugin: 0 for plugin in all_plugins}
    plugin_counts = Counter(line.Plugin for line in split_lines)
    total = sum(plugin_counts.values())
    return {plugin: plugin_counts.get(plugin, 0) / total if total else 0 for plugin in all_plugins}

# Group speakers by their primary plugin
speakers_by_plugin = defaultdict(list)
for speaker in remaining_speakers:
    speakers_by_plugin[speaker_primary_plugin[speaker]].append(speaker)

# Assign speakers to splits trying to balance plugin distribution and target ratios
remaining_speakers_set = set(remaining_speakers)
while remaining_speakers_set:
    current_train_size = len(train_lines)
    current_validation_size = len(validation_lines)
    current_test_size = len(test_lines)
    
    train_plugin_dist = get_plugin_distribution(train_lines)
    validation_plugin_dist = get_plugin_distribution(validation_lines)
    test_plugin_dist = get_plugin_distribution(test_lines)
    
    train_deficit = target_train_lines - current_train_size
    validation_deficit = target_validation_lines - current_validation_size
    test_deficit = target_test_lines - current_test_size
    
    # Skip test if test ratio is 0 and we already have test speakers from exclusives
    if TEST_RATIO01 == 0 and len(TEST_VOICES) > 0:
        test_deficit = -1
    
    if train_deficit >= validation_deficit and train_deficit >= test_deficit and train_deficit > 0:
        target_split = "train"
        target_lines = train_lines
        current_dist = train_plugin_dist
        target_remaining = target_train_lines - current_train_size
    elif validation_deficit >= test_deficit and validation_deficit > 0:
        target_split = "validation"
        target_lines = validation_lines
        current_dist = validation_plugin_dist
        target_remaining = target_validation_lines - current_validation_size
    elif test_deficit > 0:
        target_split = "test"
        target_lines = test_lines
        current_dist = test_plugin_dist
        target_remaining = target_test_lines - current_test_size
    else:
        break
    
    best_speaker = None
    best_score = -float('inf')
    
    for speaker in remaining_speakers_set:
        speaker_plugin = speaker_primary_plugin[speaker]
        plugin_score = 1.0 - current_dist.get(speaker_plugin, 0)
        speaker_line_count = len(voice_by_speaker[speaker])
        size_penalty = max(0, speaker_line_count - target_remaining) / (speaker_line_count + 1)
        score = plugin_score - size_penalty
        
        if score > best_score:
            best_score = score
            best_speaker = speaker
    
    if best_speaker:
        target_lines.extend(voice_by_speaker[best_speaker])
        print(f"Added {len(voice_by_speaker[best_speaker])} lines from {best_speaker} "
              f"({speaker_primary_plugin[best_speaker]}) to {target_split} set")
        remaining_speakers_set.remove(best_speaker)
        assigned_speakers.add(best_speaker)
    else:
        print("Error: Could not find a suitable speaker to add")
        break

# Now, collect the high-density voice lines for the test set if we have phrases
if phrases and TEST_RATIO01 > 0:
    unassigned_lines = []
    for speaker, lines in voice_by_speaker.items():
        if speaker not in assigned_speakers:
            unassigned_lines.extend(lines)
    
    if unassigned_lines:
        print(f"\nSelecting high-density phrase lines for test set...")
        line_densities = []
        for line in unassigned_lines:
            density = get_phrase_density(line, phrases)
            if density > 0:
                line_densities.append((line, density))
        line_densities.sort(key=lambda x: x[1], reverse=True)
        
        high_density_count = min(HIGH_DENSITY_TEST_LINES, len(line_densities))
        if high_density_count > 0:
            high_density_lines = [pair[0] for pair in line_densities[:high_density_count]]
            test_lines.extend(high_density_lines)
            high_density_set = set(high_density_lines)
            unassigned_lines = [line for line in unassigned_lines if line not in high_density_set]
            print(f"Added {high_density_count} high-density phrase lines to test set")
            if unassigned_lines:
                random.shuffle(unassigned_lines)
                train_count = int(len(unassigned_lines) * (TRAIN_RATIO01 / (TRAIN_RATIO01 + VALIDATION_RATIO01)))
                train_lines.extend(unassigned_lines[:train_count])
                validation_lines.extend(unassigned_lines[train_count:])
                print(f"Distributed remaining {len(unassigned_lines)} lines: "
                      f"{train_count} to train, {len(unassigned_lines) - train_count} to validation")
        else:
            print("No lines with matching phrases found for high-density selection")

# Randomize the order of lines within each split before saving
random.shuffle(train_lines)
random.shuffle(validation_lines)
random.shuffle(test_lines)
print("Randomized the order of voice lines within each split")

# Create and save the split datasets
train_dataset = VoiceLines(train_lines)
validation_dataset = VoiceLines(validation_lines)
test_dataset = VoiceLines(test_lines)

train_dataset.save_to_yaml(TRAIN_OUTPUT)
validation_dataset.save_to_yaml(VALIDATION_OUTPUT)
test_dataset.save_to_yaml(TEST_OUTPUT)

# Print summary
total = len(train_lines) + len(validation_lines) + len(test_lines)
train_plugins = Counter(line.Plugin for line in train_lines)
validation_plugins = Counter(line.Plugin for line in validation_lines)
test_plugins = Counter(line.Plugin for line in test_lines)

print(f"\nDataset split complete:")
print(f"Train: {len(train_lines)} lines ({len(train_lines)/total:.2%})")
print(f"Validation: {len(validation_lines)} lines ({len(validation_lines)/total:.2%})")
print(f"Test: {len(test_lines)} lines ({len(test_lines)/total:.2%})")
print(f"Total: {total} lines")

print("\nPlugin distribution:")
for plugin in all_plugins:
    print(f"  {plugin}:")
    t_pct = train_plugins.get(plugin, 0)/sum(train_plugins.values()) if sum(train_plugins.values()) else 0
    v_pct = validation_plugins.get(plugin, 0)/sum(validation_plugins.values()) if sum(validation_plugins.values()) else 0
    te_pct = test_plugins.get(plugin, 0)/sum(test_plugins.values()) if sum(test_plugins.values()) else 0
    print(f"    Train: {train_plugins.get(plugin, 0)} ({t_pct:.2%})")
    print(f"    Validation: {validation_plugins.get(plugin, 0)} ({v_pct:.2%})")
    print(f"    Test: {test_plugins.get(plugin, 0)} ({te_pct:.2%})")

train_speakers = set(line.VoiceType for line in train_lines)
validation_speakers = set(line.VoiceType for line in validation_lines)
test_speakers = set(line.VoiceType for line in test_lines)

print("\nSpeaker distribution:")
print(f"  Train: {len(train_speakers)} unique speakers")
print(f"  Validation: {len(validation_speakers)} unique speakers")
print(f"  Test: {len(test_speakers)} unique speakers")

train_and_val = train_speakers.intersection(validation_speakers)
train_and_test = train_speakers.intersection(test_speakers)
val_and_test = validation_speakers.intersection(test_speakers)

if not train_and_val and not train_and_test and not val_and_test:
    print("\nSuccess: No speakers appear in multiple splits")
else:
    print("\nWarning: Some speakers appear in multiple splits:")
    if train_and_val: print(f"  Train and validation: {train_and_val}")
    if train_and_test: print(f"  Train and test: {train_and_test}")
    if val_and_test: print(f"  Validation and test: {val_and_test}")

if phrases:
    train_density = sum(count_phrases_in_voice_line(line, phrases) for line in train_lines) / max(1, len(train_lines))
    val_density = sum(count_phrases_in_voice_line(line, phrases) for line in validation_lines) / max(1, len(validation_lines))
    test_density = sum(count_phrases_in_voice_line(line, phrases) for line in test_lines) / max(1, len(test_lines))
    
    print("\nPhrase density (avg phrases per voice line):")
    print(f"  Train: {train_density:.2f}")
    print(f"  Validation: {val_density:.2f}")
    print(f"  Test: {test_density:.2f}")
    
    train_with_phrases = sum(1 for line in train_lines if count_phrases_in_voice_line(line, phrases) > 0)
    val_with_phrases = sum(1 for line in validation_lines if count_phrases_in_voice_line(line, phrases) > 0)
    test_with_phrases = sum(1 for line in test_lines if count_phrases_in_voice_line(line, phrases) > 0)
    
    print("\nVoice lines containing target phrases:")
    print(f"  Train: {train_with_phrases} ({train_with_phrases/max(1, len(train_lines)):.2%})")
    print(f"  Validation: {val_with_phrases} ({val_with_phrases/max(1, len(validation_lines)):.2%})")
    print(f"  Test: {test_with_phrases} ({test_with_phrases/max(1, len(test_lines)):.2%})")

    # --- New Blocks: Output samples per phrase in YAML for training and test sets ---
    samples_per_phrase_train = {}
    for phrase in phrases:
        count = sum(1 for line in train_lines if voice_line_contains_phrase(line, phrase))
        samples_per_phrase_train[phrase] = count

    train_yaml_filename = "2-samples-per-word-train.yaml"
    with open(train_yaml_filename, "w", encoding="utf-8") as outFile:
        yaml.dump(samples_per_phrase_train, outFile, default_flow_style=False)
    print(f"\nSaved samples per phrase for training set to file: {train_yaml_filename}")

    samples_per_phrase_test = {}
    for phrase in phrases:
        count = sum(1 for line in test_lines if voice_line_contains_phrase(line, phrase))
        samples_per_phrase_test[phrase] = count

    test_yaml_filename = "2-samples-per-word-test.yaml"
    with open(test_yaml_filename, "w", encoding="utf-8") as outFile:
        yaml.dump(samples_per_phrase_test, outFile, default_flow_style=False)
    print(f"\nSaved samples per phrase for test set to file: {test_yaml_filename}")
