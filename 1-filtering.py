#!/usr/bin/env python3
import os
import sys
import re
import subprocess
from pathlib import Path
import torch
import numpy as np
import tqdm
import librosa
from Pipeline import FILTERED_VOICE_FILE_DIR, PHRASES_FILE, VoiceLines, VoiceLine, print_stage_header, VOICE_FILE_DIR
from colorama import Fore, Style

# =========== Configuration Variables ===========
# List of forbidden speakers (voice types)
FORBIDDEN_SPEAKERS = [
    "DLC1SCDragonBoneDragon", 
    "crdogvoice",
    "crdragonpriestvoice",
    "crdragonvoice",
    "crdraugrvoice",
    "crdremoravoice",
    "crdwarfenspherevoice", 
    "crdwarvenspidervoice", 
    "crfalmervoice",        
    "crhagravenvoice",
    "cruniquealduin",
    "cruniqueodahviing",
    "cruniquepaarthurnax",
    "femaleuniqueghost",
    "maleuniqueghost",
    "dlc1ld_femalenorduniquekatria",
    "dlc1maleuniquejiub",
    "dlc1maleuniquesnowelfghost",
    "dlc2crgiantvoicekarstaag",
    "dlc2maleuniquemiraak", 
    "dlc2rieklingvoice",
    "femalechild",
    "femaleuniqueazura",
    "femaleuniqueboethiah",
    "femaleuniquemephala",
    "maleuniqueclavicusvile",
    "maleuniquehermaeusmora",
    "femaleuniquemerida",
    "femaleuniquenamira",
    "femaleuniqueperiyte",
    "femaleuniquenightmother",
    "femaleuniquenocturnal",
    "femaleuniquevaermina",
    "maleuniqueaventusaretino",
    "maleuniquecicero",
    "maleuniqueblackdoor",
    "maleuniqueedbguardian",
    "maleuniquedbspectrallachance",
    "maleuniquegallus",
    "maleuniqueghostsvaknir",
    "maleuniquehircine",
    "maleuniquemalacath",
    "maleuniquemehrunesdagon",
    "maleuniquehermaeusmora",
    "maleuniquemgaugur",
    "maleuniquemolagbal",
    "maleuniqueperyite",
    "avernavoice",
    "bardvoice",
    "batheruvoice",
    "bergrisarvoice",
    "caylenevoice",
    "crdogvoice",
    "crhagravenvoice",
    "crwolfvoice",
    "dagrilonvoice", 
    "darksteelvoice",
    "femaleargonian",
    "fjonasfamiliarvoice",
    "garrettvoice",
    "gnivesvoice",
    "inarivoice",
    "iorelvoice",
    "lagduvoice",
    "lajjanvoice",
    "nasrinvoice",
    "nairvoice",
    "nelovoice",
    "qadojovoice",
    "raynesvoice",
    "rgbanditvoice",
    "saltythroatvoice",
    "cot_construct",
"cot_flies",
"cot_gk",
"cot_idara",
"cot_krellyk",
"cot_lork",
"cot_maggs",
"cot_nelly",
"cot_strahg",
"cot_tbr",
"cot_ven",
"cyrfemalchild",
"dg04groshakvoice",
"dg04molagbal",
"femaledarkelfheneri",
"frjtchiefvoice",
"frtrecruitervoice",
"jrdumzbtharvoice"
]

# Forbidden substrings that indicate low quality transcriptions
FORBIDDEN_SUBSTRINGS = [
    " ha ha ", " hmm ", " uh ", " um ", " mm ", " ah ", 
    " er ", " eh ", " oh ", " mhm ", " mm-hmm ", " mmhm ", 
    " mm-hmm ", " mm-hm ", " mm-hm "
]

# Minimum word count threshold
MIN_WORD_COUNT = 4

# Maximum transcription word count
MAX_WORD_COUNT = 135

# Maximum audio duration in seconds
MAX_AUDIO_DURATION_SEC = 29

# Silero VAD configuration (threshold value is clamped between 0 and 1, hence the suffix "01")
VAD_THRESHOLD01 = 0.7
SILENCE_THRESHOLD_START_SEC = 1.0
SILENCE_THRESHOLD_END_SEC = 1.0
SILENCE_THRESHOLD_MID_SEC = 1.0

# Data distribution balancing configuration
# For speakers: the total duration allowed per speaker is set to the total duration
# of the nth highest speaker (configurable)
SPEAKER_N = 3  
# For target words: the maximum allowed occurrences per target word
MAX_WORD_OCCURRENCES = 60

# Input and output files
INPUT_FILE = "0-parsed.yaml"
OUTPUT_FILE = "1-filtered.yaml"

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

def count_target_words(line: VoiceLine) -> int:
    """
    Count the number of target words (from PHRASES_FILE) present in the voice line's transcription.
    """
    count = 0
    try:
        with open(PHRASES_FILE, 'r', encoding='utf-8') as f:
            phrases = [l.strip() for l in f if l.strip()]
    except Exception as e:
        phrases = []
    for phrase in phrases:
        if voice_line_contains_phrase(line, phrase):
            count += 1
    return count

# ===== Updated Filter Function for Target Words =====
def filter_has_no_target_word(voice_lines: VoiceLines, original_total):
    """
    Remove voice lines that do not contain any target words based on updated normalized matching rules.
    """
    # Load target phrases
    with open(PHRASES_FILE, 'r', encoding='utf-8') as f:
        phrases = [line.strip() for line in f if line.strip()]
        
    def contains_target_word(line: VoiceLine) -> bool:
        for phrase in phrases:
            if voice_line_contains_phrase(line, phrase):
                return True
        return False
    
    filtered_lines = []
    for line in tqdm.tqdm(voice_lines.lines, desc="Filtering for target words"):
        if contains_target_word(line):
            filtered_lines.append(line)
    
    result = VoiceLines(filtered_lines)
    
    # Print total removed and percentage of original
    removed_total = original_total - len(result)
    print(f"{Fore.YELLOW}Total removed so far: {removed_total} ({removed_total / original_total:.2%} of original)")
    
    return result

# ===== Data Distribution Balancing Functions =====
def balance_speakers(voice_lines: VoiceLines) -> VoiceLines:
    """
    Balance voice lines distribution by speaker based on total duration thresholds.
    For each speaker with a total duration exceeding the threshold, remove lines
    (preferring those with fewer target words) until the total duration is below the threshold.
    """
    # Group voice lines by speaker (VoiceType)
    speakers = {}
    for line in voice_lines.lines:
        speakers.setdefault(line.VoiceType, []).append(line)
        
    # Compute total duration per speaker (in ms)
    speaker_duration = {speaker: sum(line.DurationMs for line in lines if line.DurationMs > 0)
                        for speaker, lines in speakers.items()}
    
    # Sort speakers by total duration descending
    sorted_speakers = sorted(speaker_duration.items(), key=lambda x: x[1], reverse=True)
    
    if len(sorted_speakers) >= SPEAKER_N:
        threshold = sorted_speakers[SPEAKER_N - 1][1]
    else:
        threshold = sorted_speakers[-1][1] if sorted_speakers else 0
        
    remaining_lines = []
    for speaker, lines in speakers.items():
        total_duration = sum(line.DurationMs for line in lines if line.DurationMs > 0)
        if total_duration <= threshold:
            remaining_lines.extend(lines)
        else:
            # Sort lines by target word count (ascending), then by duration descending
            sorted_lines = sorted(lines, key=lambda l: (count_target_words(l), -l.DurationMs))
            current_duration = total_duration
            kept_lines = sorted_lines.copy()
            for line in sorted_lines:
                if current_duration <= threshold:
                    break
                kept_lines.remove(line)
                current_duration -= line.DurationMs
            remaining_lines.extend(kept_lines)
    return VoiceLines(remaining_lines)

def balance_words(voice_lines: VoiceLines) -> VoiceLines:
    """
    Balance voice lines distribution by target words based on maximum allowed occurrences.
    For each target phrase that appears in more than MAX_WORD_OCCURRENCES voice lines,
    iteratively remove extra voice lines—but only if the voice line does not contain any other
    target phrase that is not yet at the limit.
    """
    # Load target phrases
    with open(PHRASES_FILE, 'r', encoding='utf-8') as f:
        target_phrases = [line.strip() for line in f if line.strip()]

    # Build mapping: voice line -> list of target phrases contained
    line_to_phrases = {}
    for line in voice_lines.lines:
        contained = []
        for phrase in target_phrases:
            if voice_line_contains_phrase(line, phrase):
                contained.append(phrase)
        line_to_phrases[line] = contained

    # Build mapping: target phrase -> set of voice lines that contain it
    phrase_to_lines = {phrase: set() for phrase in target_phrases}
    for line, phrases in line_to_phrases.items():
        for phrase in phrases:
            phrase_to_lines[phrase].add(line)

    # Initialize counts for each target phrase
    counts = {phrase: len(phrase_to_lines[phrase]) for phrase in target_phrases}

    # Start with all voice lines kept
    kept_lines = set(voice_lines.lines)

    changed = True
    while changed:
        changed = False
        # Iterate over each target phrase with a progress bar.
        for phrase in tqdm.tqdm(target_phrases, desc="Balancing target words", leave=False):
            while counts[phrase] > MAX_WORD_OCCURRENCES:
                # Candidates: lines (still kept) that contain this phrase and for which every contained target phrase
                # is already over the limit.
                candidates = []
                for line in phrase_to_lines[phrase]:
                    if line in kept_lines:
                        contained = line_to_phrases[line]
                        if all(counts[q] > MAX_WORD_OCCURRENCES for q in contained):
                            candidates.append(line)
                if not candidates:
                    break
                # Choose candidate with the fewest target phrases (i.e. contributing less to other targets)
                candidate = min(candidates, key=lambda l: len(line_to_phrases[l]))
                kept_lines.remove(candidate)
                changed = True
                for q in line_to_phrases[candidate]:
                    counts[q] -= 1
    remaining_lines = [line for line in voice_lines.lines if line in kept_lines]
    return VoiceLines(remaining_lines)

# ===== Existing Filtering Functions (unchanged) =====
def filter_missing_audios(voice_lines, original_total):
    """Remove voice lines with missing audio files."""
    result = voice_lines.filter("Missing Audio", 
                             lambda line: line.FileName != "_.wav" or os.path.exists(os.path.join(VOICE_FILE_DIR, line.FileName)))  
    
    removed_total = original_total - len(result)
    print(f"{Fore.YELLOW}Total removed so far: {removed_total} ({removed_total / original_total:.2%} of original)")
    
    return result

def filter_empty_transcriptions(voice_lines, original_total):
    """Remove voice lines with empty transcriptions."""
    result = voice_lines.filter("Empty Transcription", 
                             lambda line: line.Transcription.strip() != "")
    
    removed_total = original_total - len(result)
    print(f"{Fore.YELLOW}Total removed so far: {removed_total} ({removed_total / original_total:.2%} of original)")
    
    return result

def filter_forbidden_speakers(voice_lines, original_total):
    """Remove voice lines from forbidden speakers."""
    result = voice_lines.filter("Forbidden Speakers",
                             lambda line: line.VoiceType not in FORBIDDEN_SPEAKERS)
    
    removed_total = original_total - len(result)
    print(f"{Fore.YELLOW}Total removed so far: {removed_total} ({removed_total / original_total:.2%} of original)")
    
    return result

def filter_forbidden_substrings(voice_lines, original_total):
    """Remove voice lines containing forbidden substrings."""
    def has_no_forbidden_substrings(line):
        normalized_text = " " + line.Transcription.lower() + " "
        return all(substr not in normalized_text for substr in FORBIDDEN_SUBSTRINGS)
    
    result = voice_lines.filter("Forbidden Substrings", has_no_forbidden_substrings)
    
    removed_total = original_total - len(result)
    print(f"{Fore.YELLOW}Total removed so far: {removed_total} ({removed_total / original_total:.2%} of original)")
    
    return result

def filter_by_word_count(voice_lines, original_total):
    """Remove voice lines with word count less than or equal to MIN_WORD_COUNT or greater than MAX_WORD_COUNT."""
    def has_valid_word_count(line):
        words = re.findall(r'\b\w+\b', line.Transcription)
        return MIN_WORD_COUNT < len(words) <= MAX_WORD_COUNT
    
    result = voice_lines.filter("Word Count", has_valid_word_count)
    
    removed_total = original_total - len(result)
    print(f"{Fore.YELLOW}Total removed so far: {removed_total} ({removed_total / original_total:.2%} of original)")
    
    return result

def filter_text_in_brackets(voice_lines, original_total):
    """Remove voice lines with text in carets or parentheses."""
    def has_no_bracketed_text(line):
        return not re.search(r'<[^>]*>|\([^)]*\)', line.Transcription)
    
    result = voice_lines.filter("Bracketed Text", has_no_bracketed_text)
    
    removed_total = original_total - len(result)
    print(f"{Fore.YELLOW}Total removed so far: {removed_total} ({removed_total / original_total:.2%} of original)")
    
    return result

def filter_repetitive_words(voice_lines, original_total):
    """Remove voice lines with more than 3 consecutive repetitions of the same word."""
    def has_no_excessive_repetition(line):
        clean_text = re.sub(r'[^\w\s]', '', line.Transcription)
        words = clean_text.lower().split()
        for i in range(len(words) - 3):
            if words[i] == words[i+1] == words[i+2] == words[i+3]:
                return False
        return True
    
    result = voice_lines.filter("Repetitive Words", has_no_excessive_repetition)
    
    removed_total = original_total - len(result)
    print(f"{Fore.YELLOW}Total removed so far: {removed_total} ({removed_total / original_total:.2%} of original)")
    
    return result

def measure_audio_durations(voice_lines):
    """
    Measure and set the duration of each audio file using ffmpeg.
    """
    print(f"{Fore.CYAN}Measuring audio durations...")
    
    voice_lines_with_duration = VoiceLines()
    total_lines = len(voice_lines.lines)
    
    for idx, line in enumerate(tqdm.tqdm(voice_lines.lines)):
        if idx % 100 == 0 or idx == total_lines - 1:
            print(f"{Style.DIM}Processing voice line {idx+1}/{total_lines} ({(idx+1)/total_lines:.2%})")
        audio_path = os.path.join(VOICE_FILE_DIR, line.FileName)
        
        if not os.path.exists(audio_path):
            print(f"{Style.DIM}Warning: Audio file not found: {audio_path}")
            line.DurationMs = -1
            voice_lines_with_duration.add(line)
            continue
            
        try:
            cmd_info = [
                'ffprobe', 
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                audio_path
            ]
            
            result = subprocess.run(cmd_info, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            duration_sec = float(result.stdout.strip())
            line.DurationMs = int(duration_sec * 1000)
            
            voice_lines_with_duration.add(line)
            
        except Exception as e:
            print(f"{Fore.RED}Error measuring duration for {audio_path}: {str(e)}")
            line.DurationMs = -1
            voice_lines_with_duration.add(line)
    
    return voice_lines_with_duration

def filter_by_audio_duration(voice_lines, original_total):
    """Remove voice lines with audio longer than MAX_AUDIO_DURATION_SEC seconds."""
    result = voice_lines.filter("Audio Duration",
                             lambda line: line.DurationMs > 0 and line.DurationMs <= MAX_AUDIO_DURATION_SEC * 1000)
    
    removed_total = original_total - len(result)
    print(f"{Fore.YELLOW}Total removed so far: {removed_total} ({removed_total / original_total:.2%} of original)")
    
    return result

def filter_by_silero_vad(voice_lines, original_total):
    """
    Filter out voice lines based on speech detection using Silero VAD.
    Removes lines with:
    - No detected speech
    - Excessive silence at the start, middle, or end
    """
    print(f"{Fore.CYAN}Running Silero VAD speech detection...")
    
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False)
    
    (get_speech_timestamps, _, _, _, _) = utils
    
    filtered_lines = VoiceLines()
    filtered_lines.original_count = voice_lines.original_count
    
    rejected_count = 0
    no_speech_count = 0
    start_silence_count = 0
    end_silence_count = 0
    mid_silence_count = 0
    
    total_lines = len(voice_lines.lines)
    
    for idx, line in enumerate(tqdm.tqdm(voice_lines.lines)):
        if idx % 100 == 0 or idx == total_lines - 1:
            print(f"{Style.DIM}Processing voice line {idx+1}/{total_lines} ({(idx+1)/total_lines:.2%})")
        audio_path = os.path.join(VOICE_FILE_DIR, line.FileName)
        
        if not os.path.exists(audio_path):
            filtered_lines.add(line)
            continue
        
        try:
            audio, sample_rate = librosa.load(audio_path, sr=16000)
            if not isinstance(audio, torch.Tensor):
                audio = torch.FloatTensor(audio)
            
            speech_timestamps = get_speech_timestamps(
                audio, model, threshold=VAD_THRESHOLD01, 
                sampling_rate=sample_rate
            )
            
            if not speech_timestamps:
                no_speech_count += 1
                rejected_count += 1
                continue
            
            total_duration_sec = len(audio) / sample_rate
            speech_start_sec = speech_timestamps[0]['start'] / sample_rate
            speech_end_sec = speech_timestamps[-1]['end'] / sample_rate
            
            if speech_start_sec > SILENCE_THRESHOLD_START_SEC:
                start_silence_count += 1
                rejected_count += 1
                continue
            
            if total_duration_sec - speech_end_sec > SILENCE_THRESHOLD_END_SEC:
                end_silence_count += 1
                rejected_count += 1
                continue
            
            has_long_silence = False
            for i in range(len(speech_timestamps) - 1):
                current_end = speech_timestamps[i]['end'] / sample_rate
                next_start = speech_timestamps[i + 1]['start'] / sample_rate
                if next_start - current_end > SILENCE_THRESHOLD_MID_SEC:
                    has_long_silence = True
                    break
            
            if has_long_silence:
                mid_silence_count += 1
                rejected_count += 1
                continue
            
            filtered_lines.add(line)
            
        except Exception as e:
            print(f"{Fore.RED}Error processing {audio_path}: {str(e)}")
            filtered_lines.add(line)
    
    print(f"{Fore.YELLOW}Silero VAD filtering results:")
    print(f"{Style.DIM}  No speech detected: {no_speech_count}")
    print(f"{Style.DIM}  Excessive silence at start: {start_silence_count}")
    print(f"{Style.DIM}  Excessive silence at end: {end_silence_count}")
    print(f"{Style.DIM}  Excessive silence in middle: {mid_silence_count}")
    print(f"{Fore.YELLOW}Total removed by VAD: {rejected_count}")
    
    removed_total = original_total - len(filtered_lines)
    print(f"{Fore.YELLOW}Total removed so far: {removed_total} ({removed_total / original_total:.2%} of original)")
    
    return filtered_lines

def copy_audio_files(voice_lines):
    """Copy audio files from the filtered voice lines to a new directory."""
    print(f"{Fore.CYAN}Copying audio files...")
    
    output_dir = Path(FILTERED_VOICE_FILE_DIR)
    output_dir.mkdir(exist_ok=True)
    
    total_lines = len(voice_lines.lines)
    
    for idx, line in enumerate(tqdm.tqdm(voice_lines.lines)):
        if idx % 100 == 0 or idx == total_lines - 1:
            print(f"{Style.DIM}Copying audio file {idx+1}/{total_lines} ({(idx+1)/total_lines:.2%})")
        audio_path = os.path.join(VOICE_FILE_DIR, line.FileName)
        
        if not os.path.exists(audio_path):
            print(f"{Style.DIM}Warning: Audio file not found: {audio_path}")
            continue
        
        output_path = output_dir / line.FileName
        try:
            Path(audio_path).replace(output_path)
        except Exception as e:
            print(f"{Fore.RED}Error copying {audio_path} to {output_path}: {str(e)}")
    
    print(f"{Fore.GREEN}Copied {total_lines} audio files to {output_dir}")
    
    return output_dir

# ===== Main Pipeline Execution =====
def main():
    print_stage_header("Stage 1: Pre-Filtering")
    
    voice_lines = VoiceLines.load_from_yaml(INPUT_FILE)
    initial_count = len(voice_lines)
    
    # Randomize the order of voicelines before filtering
    print(f"{Fore.CYAN}Randomizing voicelines order...")
    np.random.shuffle(voice_lines.lines)
    
    print(f"Starting filtering with {initial_count} voice lines")
    
    original_total = len(voice_lines)
    
    # voice_lines = filter_empty_transcriptions(voice_lines, original_total)
    # voice_lines = filter_forbidden_speakers(voice_lines, original_total)
    # voice_lines = filter_forbidden_substrings(voice_lines, original_total)
    # voice_lines = filter_by_word_count(voice_lines, original_total)
    # voice_lines = filter_text_in_brackets(voice_lines, original_total)
    # voice_lines = filter_repetitive_words(voice_lines, original_total)
    voice_lines = filter_missing_audios(voice_lines, original_total)
    # voice_lines = filter_has_no_target_word(voice_lines, original_total)
    
    #voice_lines = measure_audio_durations(voice_lines)
    #voice_lines = filter_by_audio_duration(voice_lines, original_total)
    #voice_lines = filter_by_silero_vad(voice_lines, original_total)
    
    # Apply data distribution balancing: first speakers, then words
    # print(f"{Fore.CYAN}Balancing distribution by speaker...")
    # voice_lines = balance_speakers(voice_lines)
    print(f"{Fore.CYAN}Balancing distribution by target words...")
    #voice_lines = balance_words(voice_lines)
    
    remaining_count = len(voice_lines)
    removed_count = initial_count - remaining_count
    
    print(f"\n{Fore.YELLOW}===== Filtering Summary =====")
    print(f"{Fore.YELLOW}Initial count: {initial_count}")
    print(f"{Fore.YELLOW}Remaining count: {remaining_count}")
    print(f"{Fore.YELLOW}Total removed: {removed_count} ({removed_count / initial_count:.2%})")
    
    voice_lines.save_to_yaml(OUTPUT_FILE)
    
    copy_audio_files(voice_lines)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
