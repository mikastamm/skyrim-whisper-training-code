#!/usr/bin/env python3
"""
This script selects a random sample of voice lines for each speaker and copies them 
to a designated folder for manual review. 
"""
import os
import random
import shutil
import yaml

# ANSI escape sequences for colored console output
COLOR_IMPORTANT = "\033[93m"  # Bright yellow for important information
COLOR_MUTED = "\033[90m"      # Gray for less important details
COLOR_RESET = "\033[0m"       # Reset to default color

# Configuration variables
PREVIEW_OUTPUT_DIR = "speaker_preview"   # Destination folder for preview copies.
INPUT_YAML = "1-filtered.yaml"             # YAML file produced by Stage 0 CSV Parsing.
NUM_PREVIEW_FILES = 1                    # Number of files to select per speaker.

# Import shared classes from Pipeline.py
from Pipeline import VoiceLine, VoiceLines, VOICE_FILE_DIR

def load_voice_lines(yaml_file: str) -> VoiceLines:
    """
    Loads voice lines from a YAML file and returns a VoiceLines object.
    Assumes the YAML file contains a list of dictionaries with keys matching VoiceLine attributes.
    """
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    voice_lines = VoiceLines()
    for item in data['lines']:
        vl = VoiceLine(
            item.get("InternalFileName", ""),
            item.get("Transcription", ""),
            item.get("VoiceType", ""),
            item.get("Plugin", ""),
            item.get("State", ""),
            item.get("FileName", ""),
            item.get("DurationMs", -1),
            item.get("Stage2Data", None)
        )
        if vl.FileName == "_.wav":
            continue
        voice_lines.lines.append(vl)
    return voice_lines

def main():
    print(f"{COLOR_IMPORTANT}Starting Speaker Preview Utility{COLOR_RESET}")
    print("Loading voice lines from YAML...")
    voice_lines = load_voice_lines(INPUT_YAML)
    total_lines = len(voice_lines.lines)
    print(f"Loaded {total_lines} voice lines.")

    # Group voice lines by speaker (using VoiceType as the speaker identifier)
    speakers = {}
    for vl in voice_lines.lines:
        speaker = vl.VoiceType
        if speaker not in speakers:
            speakers[speaker] = []
        speakers[speaker].append(vl)

    # Create output directory if it does not exist
    if not os.path.exists(PREVIEW_OUTPUT_DIR):
        os.makedirs(PREVIEW_OUTPUT_DIR)
        print(f"{COLOR_MUTED}Created output directory: {PREVIEW_OUTPUT_DIR}{COLOR_RESET}")

    # Process each speaker group
    for speaker, lines in speakers.items():
        num_to_select = min(NUM_PREVIEW_FILES, len(lines))
        selected_lines = random.sample(lines, num_to_select)
        print(f"{COLOR_IMPORTANT}Speaker: {speaker}{COLOR_RESET}")
        print(f"Selected {num_to_select} file(s) for preview.")
        for vl in selected_lines:
            # Concatenate VOICE_FILE_DIR with the FileName to get the full path
            src_filepath = os.path.join(VOICE_FILE_DIR, vl.FileName)
            # Destination file name format: speakername_InternalFileName.wav
            dest_filename = f"{speaker}_{vl.InternalFileName}.wav"
            dest_filepath = os.path.join(PREVIEW_OUTPUT_DIR, dest_filename)
            try:
                shutil.copy(src_filepath, dest_filepath)
                print(f"Copied: {src_filepath} -> {dest_filepath}")
            except Exception as e:
                print(f"Error copying file {src_filepath}: {e}")

    print(f"{COLOR_IMPORTANT}Speaker preview generation completed.{COLOR_RESET}")

if __name__ == "__main__":
    main()

