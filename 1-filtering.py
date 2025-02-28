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

# Silero VAD configuration
VAD_THRESHOLD = 0.7  # Speech detection confidence threshold
SILENCE_THRESHOLD_START_SEC = 1.0  # Max acceptable silence at start
SILENCE_THRESHOLD_END_SEC = 1.0    # Max acceptable silence at end
SILENCE_THRESHOLD_MID_SEC = 1.0    # Max acceptable silence in middle

# Input and output files
INPUT_FILE = "0-parsed.yaml"
INTERMEDIATE_FILE = "1.1-measured.yaml"
OUTPUT_FILE = "1-filtered.yaml"

def main():
    print_stage_header("Stage 1: Pre-Filtering")
    
    # Load voice lines from previous stage
    voice_lines = VoiceLines.load_from_yaml(INPUT_FILE)
    initial_count = len(voice_lines)
    
    print(f"Starting filtering with {initial_count} voice lines")
    
    # Apply filters and track original count
    original_total = len(voice_lines)
    
    voice_lines = filter_empty_transcriptions(voice_lines, original_total)
    voice_lines = filter_forbidden_speakers(voice_lines, original_total)
    voice_lines = filter_forbidden_substrings(voice_lines, original_total)
    voice_lines = filter_by_word_count(voice_lines, original_total)
    voice_lines = filter_text_in_brackets(voice_lines, original_total)
    voice_lines = filter_repetitive_words(voice_lines, original_total)
    voice_lines = filter_missing_audios(voice_lines, original_total)
    voice_lines = filter_has_no_target_word(voice_lines, original_total)
    # Disabled - takes too long
    if os.path.exists(INTERMEDIATE_FILE):
        print(f"{Fore.GREEN}Found existing measurement file {INTERMEDIATE_FILE}, loading instead of remeasuring...")
        voice_lines = VoiceLines.load_from_yaml(INTERMEDIATE_FILE)
    else:
        voice_lines = measure_audio_durations(voice_lines)
        # Save intermediate file after measuring
        print(f"{Fore.GREEN}Saving measured voice lines to {INTERMEDIATE_FILE}")
        voice_lines.save_to_yaml(INTERMEDIATE_FILE)
     
    voice_lines = filter_by_audio_duration(voice_lines, original_total)
    voice_lines = filter_by_silero_vad(voice_lines, original_total)
    
    # Print summary of filtering operations
    remaining_count = len(voice_lines)
    removed_count = initial_count - remaining_count
    
    print(f"\n{Fore.YELLOW}===== Filtering Summary =====")
    print(f"{Fore.YELLOW}Initial count: {initial_count}")
    print(f"{Fore.YELLOW}Remaining count: {remaining_count}")
    print(f"{Fore.YELLOW}Total removed: {removed_count} ({removed_count / initial_count:.2%})")
    
    # Save filtered voice lines
    voice_lines.save_to_yaml(OUTPUT_FILE)
    
    # Copy audio files to a new directory
    copy_audio_files(voice_lines)
    
    return 0

def copy_audio_files(voice_lines):
    """Copy audio files from the filtered voice lines to a new directory."""
    print(f"{Fore.CYAN}Copying audio files...")
    
    # Create output directory if it doesn't exist
    output_dir = Path(FILTERED_VOICE_FILE_DIR)
    output_dir.mkdir(exist_ok=True)
    
    total_lines = len(voice_lines.lines)
    
    for idx, line in enumerate(tqdm.tqdm(voice_lines.lines)):
        # Log progress periodically
        if idx % 100 == 0 or idx == total_lines - 1:
            print(f"{Style.DIM}Copying audio file {idx+1}/{total_lines} ({(idx+1)/total_lines:.2%})")
        audio_path = os.path.join(VOICE_FILE_DIR, line.FileName)
        
        # Skip if the file doesn't exist
        if not os.path.exists(audio_path):
            print(f"{Style.DIM}Warning: Audio file not found: {audio_path}")
            continue
        
        # Copy the file to the output directory
        output_path = output_dir / line.FileName
        try:
            Path(audio_path).replace(output_path)
        except Exception as e:
            print(f"{Fore.RED}Error copying {audio_path} to {output_path}: {str(e)}")
    
    print(f"{Fore.GREEN}Copied {total_lines} audio files to {output_dir}")
    
    return output_dir

def filter_has_no_target_word(voice_lines:VoiceLines, original_total):
    # Load the words file
    with open(PHRASES_FILE, 'r', encoding='utf-8') as f:
        phrase_texts = [line.strip() for line in f if line.strip()]
        
    # Convert to lowercase for case-insensitive matching
    phrase_texts_lower = [phrase.lower() for phrase in phrase_texts]
    
    # Filter out voice lines that do not contain any target words
    def contains_target_word(line):
        transcription_lower = line.Transcription.lower()
        return any(phrase in transcription_lower for phrase in phrase_texts_lower)
    
    result = voice_lines.filter("No Target Word", contains_target_word)
    
    # Print total removed and percentage of original
    removed_total = original_total - len(result)
    print(f"{Fore.YELLOW}Total removed so far: {removed_total} ({removed_total / original_total:.2%} of original)")
    
    return result
    

def filter_missing_audios(voice_lines, original_total):
    """Remove voice lines with missing audio files."""
    result = voice_lines.filter("Missing Audio", 
                             lambda line: line.FileName != "_.wav" and os.path.exists(os.path.join(VOICE_FILE_DIR, line.FileName)))
    
    # Print total removed and percentage of original
    removed_total = original_total - len(result)
    print(f"{Fore.YELLOW}Total removed so far: {removed_total} ({removed_total / original_total:.2%} of original)")
    
    return result
def filter_empty_transcriptions(voice_lines, original_total):
    """Remove voice lines with empty transcriptions."""
    result = voice_lines.filter("Empty Transcription", 
                             lambda line: line.Transcription.strip() != "")
    
    # Print total removed and percentage of original
    removed_total = original_total - len(result)
    print(f"{Fore.YELLOW}Total removed so far: {removed_total} ({removed_total / original_total:.2%} of original)")
    
    return result

def filter_forbidden_speakers(voice_lines, original_total):
    """Remove voice lines from forbidden speakers."""
    result = voice_lines.filter("Forbidden Speakers",
                             lambda line: line.VoiceType not in FORBIDDEN_SPEAKERS)
    
    # Print total removed and percentage of original
    removed_total = original_total - len(result)
    print(f"{Fore.YELLOW}Total removed so far: {removed_total} ({removed_total / original_total:.2%} of original)")
    
    return result

def filter_forbidden_substrings(voice_lines, original_total):
    """Remove voice lines containing forbidden substrings."""
    def has_no_forbidden_substrings(line):
        normalized_text = " " + line.Transcription.lower() + " "  # Add spaces for boundary matching
        return all(substr not in normalized_text for substr in FORBIDDEN_SUBSTRINGS)
    
    result = voice_lines.filter("Forbidden Substrings", has_no_forbidden_substrings)
    
    # Print total removed and percentage of original
    removed_total = original_total - len(result)
    print(f"{Fore.YELLOW}Total removed so far: {removed_total} ({removed_total / original_total:.2%} of original)")
    
    return result

def filter_by_word_count(voice_lines, original_total):
    """Remove voice lines with word count less than or equal to MIN_WORD_COUNT or greater than MAX_WORD_COUNT."""
    def has_valid_word_count(line):
        words = re.findall(r'\b\w+\b', line.Transcription)
        return MIN_WORD_COUNT < len(words) <= MAX_WORD_COUNT
    
    result = voice_lines.filter("Word Count", has_valid_word_count)
    
    # Print total removed and percentage of original
    removed_total = original_total - len(result)
    print(f"{Fore.YELLOW}Total removed so far: {removed_total} ({removed_total / original_total:.2%} of original)")
    
    return result

def filter_text_in_brackets(voice_lines, original_total):
    """Remove voice lines with text in carets or parentheses."""
    def has_no_bracketed_text(line):
        return not re.search(r'<[^>]*>|\([^)]*\)', line.Transcription)
    
    result = voice_lines.filter("Bracketed Text", has_no_bracketed_text)
    
    # Print total removed and percentage of original
    removed_total = original_total - len(result)
    print(f"{Fore.YELLOW}Total removed so far: {removed_total} ({removed_total / original_total:.2%} of original)")
    
    return result

def filter_repetitive_words(voice_lines, original_total):
    """Remove voice lines with more than 3 consecutive repetitions of the same word."""
    def has_no_excessive_repetition(line):
        # Remove special characters except whitespace
        clean_text = re.sub(r'[^\w\s]', '', line.Transcription)
        words = clean_text.lower().split()
        
        for i in range(len(words) - 3):
            if words[i] == words[i+1] == words[i+2] == words[i+3]:
                return False
        return True
    
    result = voice_lines.filter("Repetitive Words", has_no_excessive_repetition)
    
    # Print total removed and percentage of original
    removed_total = original_total - len(result)
    print(f"{Fore.YELLOW}Total removed so far: {removed_total} ({removed_total / original_total:.2%} of original)")
    
    return result

def measure_audio_durations(voice_lines):
    """
    Measure and set the duration of each audio file using ffmpeg.
    Returns a new VoiceLines object with updated duration values.
    """
    print(f"{Fore.CYAN}Measuring audio durations...")
    
    voice_lines_with_duration = VoiceLines()
    total_lines = len(voice_lines.lines)
    
    for idx, line in enumerate(tqdm.tqdm(voice_lines.lines)):
        # Log progress periodically
        if idx % 100 == 0 or idx == total_lines - 1:
            print(f"{Style.DIM}Processing voice line {idx+1}/{total_lines} ({(idx+1)/total_lines:.2%})")
        audio_path = os.path.join(VOICE_FILE_DIR, line.FileName)
        
        # Check if file exists
        if not os.path.exists(audio_path):
            print(f"{Style.DIM}Warning: Audio file not found: {audio_path}")
            line.DurationMs = -1
            voice_lines_with_duration.add(line)
            continue
            
        try:
            # Use ffmpeg to get duration in milliseconds
            cmd = [
                'ffmpeg', 
                '-i', audio_path, 
                '-hide_banner',
                '-v', 'error',
                '-f', 'null',
                '-'
            ]
            
            # First run to get file info
            cmd_info = [
                'ffprobe', 
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                audio_path
            ]
            
            # Run the command
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
    
    # Print total removed and percentage of original
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
    
    # Load the Silero VAD model
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
        # Log progress periodically
        if idx % 100 == 0 or idx == total_lines - 1:
            print(f"{Style.DIM}Processing voice line {idx+1}/{total_lines} ({(idx+1)/total_lines:.2%})")
        audio_path = os.path.join(VOICE_FILE_DIR, line.FileName)
        
        # Skip if the file doesn't exist
        if not os.path.exists(audio_path):
            filtered_lines.add(line)
            continue
        
        try:
            # Load audio
            audio, sample_rate = librosa.load(audio_path, sr=16000)
            
            # Ensure audio is torch tensor
            if not isinstance(audio, torch.Tensor):
                audio = torch.FloatTensor(audio)
            
            # Get speech timestamps
            speech_timestamps = get_speech_timestamps(
                audio, model, threshold=VAD_THRESHOLD, 
                sampling_rate=sample_rate
            )
            
            # Skip if no speech detected
            if not speech_timestamps:
                no_speech_count += 1
                rejected_count += 1
                continue
            
            # Calculate duration in seconds
            total_duration_sec = len(audio) / sample_rate
            
            # Get start and end of speech
            speech_start_sec = speech_timestamps[0]['start'] / sample_rate
            speech_end_sec = speech_timestamps[-1]['end'] / sample_rate
            
            # Check for silence at start
            if speech_start_sec > SILENCE_THRESHOLD_START_SEC:
                start_silence_count += 1
                rejected_count += 1
                continue
            
            # Check for silence at end
            if total_duration_sec - speech_end_sec > SILENCE_THRESHOLD_END_SEC:
                end_silence_count += 1
                rejected_count += 1
                continue
            
            # Check for long gaps between speech segments
            has_long_silence = False
            for i in range(len(speech_timestamps) - 1):
                current_end = speech_timestamps[i]['end'] / sample_rate
                next_start = speech_timestamps[i + 1]['start'] / sample_rate
                silence_duration = next_start - current_end
                
                if silence_duration > SILENCE_THRESHOLD_MID_SEC:
                    has_long_silence = True
                    break
            
            if has_long_silence:
                mid_silence_count += 1
                rejected_count += 1
                continue
            
            # If all checks pass, add the line
            filtered_lines.add(line)
            
        except Exception as e:
            print(f"{Fore.RED}Error processing {audio_path}: {str(e)}")
            # Keep the line in case of processing error
            filtered_lines.add(line)
    
    # Print detailed stats
    print(f"{Fore.YELLOW}Silero VAD filtering results:")
    print(f"{Style.DIM}  No speech detected: {no_speech_count}")
    print(f"{Style.DIM}  Excessive silence at start: {start_silence_count}")
    print(f"{Style.DIM}  Excessive silence at end: {end_silence_count}")
    print(f"{Style.DIM}  Excessive silence in middle: {mid_silence_count}")
    print(f"{Fore.YELLOW}Total removed by VAD: {rejected_count}")
    
    # Print total removed and percentage of original
    removed_total = original_total - len(filtered_lines)
    print(f"{Fore.YELLOW}Total removed so far: {removed_total} ({removed_total / original_total:.2%} of original)")
    
    return filtered_lines

if __name__ == "__main__":
    sys.exit(main())
