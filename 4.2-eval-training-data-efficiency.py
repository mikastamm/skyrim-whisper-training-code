#!/usr/bin/env python3

"""
4.2-eval-training-data-efficiency.py

Used to evaluate how word recognition efficiency changes with more examples for that phrase.
You have to run 4.1-eval-test-set.py first to generate the eval YAML files.
"""

import os
import re
import yaml
import numpy as np
from tqdm import tqdm
from Pipeline import VoiceLines, VoiceLine, PHRASES_FILE

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

def count_phrase_occurrences(training_yaml: str, phrases: list) -> dict:
    """
    Load training voicelines from training_yaml and count occurrences of each phrase.
    """
    voice_lines = VoiceLines.load_from_yaml(training_yaml)
    phrase_counts = {phrase: 0 for phrase in phrases}
    for line in tqdm(voice_lines.lines, desc="Counting phrase occurrences", unit="line"):
        for phrase in phrases:
            if voice_line_contains_phrase(line, phrase):
                phrase_counts[phrase] += 1
    return phrase_counts

def load_transcription_results(file_path: str) -> list:
    """
    Load evaluation transcription results from a YAML file.
    Expected structure: a dict with key "transcriptions" that holds a list of dicts,
    each with keys "base", "tune", and "gtgt".
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("transcriptions", [])

def compute_phrase_errors(transcription_results: list, phrases: list, mode: str) -> dict:
    """
    Compute error rates for each phrase from a list of transcription results.
    mode: "base" or "tune" (which field to check in each result).
    For each voice line, only if the ground truth contains the phrase (using normalized matching),
    error is 0 if the phrase is found in the transcription; otherwise, error is 100.
    Returns a dict mapping phrase -> list of error values.
    """
    phrase_errors = {phrase: [] for phrase in phrases}
    for result in tqdm(transcription_results, desc=f"Computing phrase errors ({mode})", unit="line"):
        gt = result.get("gtgt", "")
        norm_gt = normalize_text(gt)
        for phrase in phrases:
            norm_phrase = normalize_text(phrase)
            if norm_phrase in norm_gt:  # only evaluate if phrase appears in ground truth
                transcription = result.get(mode, "")
                norm_trans = normalize_text(transcription)
                error = 0.0 if norm_phrase in norm_trans else 100.0
                phrase_errors[phrase].append(error)
    return phrase_errors

def average_error(error_list: list) -> float:
    if error_list:
        return float(np.mean(error_list))
    else:
        return None

def main():
    # Load target phrases.
    with open(PHRASES_FILE, "r", encoding="utf-8") as f:
        target_phrases = [line.strip() for line in f if line.strip()]
    
    # Count phrase occurrences in training data ("2-train.yaml")
    training_yaml = "2-train.yaml"
    print("Counting phrase occurrences in training data...")
    phrase_counts = count_phrase_occurrences(training_yaml, target_phrases)
    
    # Configurable list of checkpoint result YAML files.
    checkpoint_files = [
        "./whisper-skyrim-en/checkpoint-1800",
        "./whisper-skyrim-en/checkpoint-1200",
        ]
    
    # Load transcription results for each checkpoint.
    results_by_ckpt = {}
    for ckpt_file in checkpoint_files:
        print(f"Loading results from {ckpt_file}...")
        results_by_ckpt[ckpt_file] = load_transcription_results(ckpt_file)
    
    # Compute baseline errors from the "base" field using the first checkpoint's results.
    baseline_results = results_by_ckpt[checkpoint_files[0]]
    errors_base = compute_phrase_errors(baseline_results, target_phrases, mode="base")
    
    # For each checkpoint, compute tuned errors.
    tuned_errors = {}  # key: checkpoint file name, value: dict of phrase->list of errors
    for ckpt_file in checkpoint_files:
        print(f"Computing tuned errors for {ckpt_file}...")
        tuned_errors[ckpt_file] = compute_phrase_errors(results_by_ckpt[ckpt_file], target_phrases, mode="tune")
    
    # Build phrase info: for each phrase, store training count, baseline error, and tuned errors for each checkpoint.
    phrase_info = {}
    for phrase in target_phrases:
        base_err = average_error(errors_base.get(phrase, []))
        info = {
            "training_count": phrase_counts.get(phrase, 0),
            "baseline_error": base_err
        }
        for ckpt_file in checkpoint_files:
            ckpt_name = os.path.splitext(os.path.basename(ckpt_file))[0]  # e.g. "transcription_results_checkpoint-1200"
            info[f"error_{ckpt_name}"] = average_error(tuned_errors[ckpt_file].get(phrase, []))
        phrase_info[phrase] = info

    # Filter out phrases that have a training count of 0 or were never found in evaluation data (baseline_error is None).
    phrase_info = {phrase: info for phrase, info in phrase_info.items() 
                   if info["training_count"] != 0 and info["baseline_error"] is not None}

    # Bin phrases by training count.
    phrases_sorted = sorted(phrase_info.items(), key=lambda item: item[1]["training_count"], reverse=True)
    n = len(phrases_sorted)
    bin1 = phrases_sorted[: n // 4]           # Top 25%
    bin2 = phrases_sorted[n // 4: n // 2]        # Next 25%
    bin3 = phrases_sorted[n // 2: 3 * n // 4]    # Next 25 (i.e. phrases ranked between 50–75%)
    bin4 = phrases_sorted[3 * n // 4:]           # Bottom 25%

    # Function to compute bin score for a given bin and checkpoint.
    def bin_score(bin_data, checkpoint_key):
        total_delta = 0.0
        total_count = 0
        for phrase, info in bin_data:
            base_err = info["baseline_error"]
            tuned_err = info.get(checkpoint_key)
            count = info["training_count"]
            if base_err is not None and tuned_err is not None and count > 0:
                total_delta += (base_err - tuned_err)
                total_count += count
        return total_delta / total_count if total_count > 0 else None

    # For each checkpoint, compute bin scores.
    bin_scores = {}  # key: checkpoint key, value: dict of bin scores
    # Also compute for all data (using each checkpoint)
    for ckpt_file in checkpoint_files:
        ckpt_name = os.path.splitext(os.path.basename(ckpt_file))[0]
        checkpoint_key = f"error_{ckpt_name}"
        total_delta = 0.0
        total_count = 0
        for phrase, info in phrase_info.items():
            base_err = info["baseline_error"]
            tuned_err = info.get(checkpoint_key)
            count = info["training_count"]
            if base_err is not None and tuned_err is not None and count > 0:
                total_delta += (base_err - tuned_err)
                total_count += count
        all_data_score = total_delta / total_count if total_count > 0 else None
        bin_scores[ckpt_name] = {
            "all_data": all_data_score,
            "top_25": bin_score(bin1, checkpoint_key),
            "next_25": bin_score(bin2, checkpoint_key),
            "next_25_after_top_50": bin_score(bin3, checkpoint_key),
            "bottom_25": bin_score(bin4, checkpoint_key)
        }

    # Prepare final output.
    output = {
        "bin_scores": bin_scores,
        "phrase_info": phrase_info
    }
    
    output_file = "4.2-eval-training-data-efficiency.yaml"
    with open(output_file, "w", encoding="utf-8") as f:
        yaml.dump(output, f, sort_keys=False, allow_unicode=True)
    print(f"Phrase analysis written to {output_file}")

if __name__ == "__main__":
    main()
