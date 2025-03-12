#!/usr/bin/env python
"""
This file selects training data for phrase-based speech training.
It loads filtered voice lines from a YAML file, checks for phrase occurrence
using normalized matching, and then selects voice lines based on configurable
criteria (minimum, target, and maximum appearances per phrase). The selected
training data (as a count and list of voice lines) is then saved to disk, along
with details about any discarded phrases as a list of voice lines.
"""
  
# ----------------------- Configurable Constants -----------------------
# Filtering parameters for training data selection.
MIN_APPEARANCES = 5       # Minimum number of voice lines required for a phrase.
TARGET_APPEARANCES = 40    # Target number of voice lines to aim for per phrase.
MAX_APPEARANCES = 70      # Maximum number of voice lines allowed per phrase.
WORDS_FILE = "words.txt"  # File containing words (used to check existence).
INPUT_FILE = "1-filtered.yaml"  # YAML file to load filtered voice lines.
OUTPUT_TRAINING_DATA = "2-training-data.yaml"  # Output YAML for selected training data.
OUTPUT_DISCARDED = "2-discarded.yaml"  # Output YAML for discarded phrases.
# ----------------------------------------------------------------------
  
import yaml
import random
import os
import re
import sys
from typing import List, Dict, Any, Set, Optional, Tuple, Counter
from collections import defaultdict, Counter
from enum import Enum, auto
  
from Pipeline import VoiceLine, VoiceLines, print_stage_header, VOICE_FILE_DIR, PHRASES_FILE
  
# --- New normalization and phrase-checking functions ---
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
  
# --- Existing Classes and Enums ---
class PhraseDetermination:
    """Intermediate variables for the phrase selection process."""
    def __init__(self):
        self.selected_voice_lines: List[VoiceLine] = []
        self.speakers: Set[str] = set()  # Track unique speakers
        self.speaker_counts: Counter = Counter()  # Track counts per speaker
        self.punctuation_types: Dict[str, bool] = {
            ',': False,
            '.': False,
            '!': False,
            '?': False
        }
        self.sentence_positions: Dict[str, int] = {
            'start': 0,
            'middle': 0,
            'end': 0
        }
        self.discard_reason: Optional[str] = None
  
class SentencePosition(Enum):
    START = auto()
    MIDDLE = auto()
    END = auto()
  
class Phrase:
    """Represents a phrase and its associated voice lines for training data selection."""
    def __init__(self, text: str):
        self.text: str = text
        self.voice_lines: List[VoiceLine] = []
        self.determining: PhraseDetermination = PhraseDetermination()
    @property
    def total_duration_ms(self) -> int:
        """Calculate total duration of all voice lines for this phrase in milliseconds."""
        return sum(vl.DurationMs for vl in self.voice_lines)
    def to_dict(self) -> Dict[str, Any]:
        """Convert the Phrase to a dictionary for YAML serialization (without determining)."""
        return {
            "text": self.text,
            "voice_lines": [vl.to_dict() for vl in self.voice_lines],
            "total_duration_ms": self.total_duration_ms
        }
  
class SelectedTrainingData:
    """Container for selected training data."""
    def __init__(self):
        self.phrases: List[Phrase] = []
        self.voice_lines: List[VoiceLine] = []
    @property
    def duration_minutes(self) -> float:
        """Calculate total duration of all voice lines in minutes."""
        total_ms = sum(vl.DurationMs for vl in self.voice_lines)
        return total_ms / 60000  # Convert from ms to minutes
    def save_to_yaml(self, file_path: str) -> None:
        """
        Save the selected training data to a YAML file with the following format:
        count: <number of voice lines>
        lines: [list of voice line dictionaries]
        """
        data = {
            "count": len(self.voice_lines),
            "lines": [vl.to_dict() for vl in self.voice_lines]
        }
        with open(file_path, 'w', encoding='utf-8') as file:
            yaml.dump(data, file, sort_keys=False, default_flow_style=False)
        print(f"Saved selected training data with {len(self.voice_lines)} voice lines to {file_path}")
  
# --- Helper Functions for Sentence Position and Punctuation ---
def determine_sentence_position(transcription: str, phrase: str) -> SentencePosition:
    """Determine the position of the phrase within the transcription."""
    clean_transcription = re.sub(r'[^\w\s]', '', transcription.lower())
    clean_phrase = re.sub(r'[^\w\s]', '', phrase.lower())
    phrase_index = clean_transcription.find(clean_phrase)
    if phrase_index == -1:
        return SentencePosition.MIDDLE
    total_length = len(clean_transcription)
    phrase_end_index = phrase_index + len(clean_phrase)
    if phrase_index < total_length * 0.25:
        return SentencePosition.START
    elif phrase_end_index > total_length * 0.75:
        return SentencePosition.END
    else:
        return SentencePosition.MIDDLE
  
def get_ending_punctuation(transcription: str, phrase: str) -> Optional[str]:
    """Get the punctuation mark following the phrase in the transcription."""
    clean_phrase = phrase.lower().strip()
    transcription_lower = transcription.lower()
    index = transcription_lower.find(clean_phrase)
    if index == -1:
        return None
    end_idx = index + len(clean_phrase)
    if end_idx < len(transcription) and transcription[end_idx] in ',.!?':
        return transcription[end_idx]
    return None
  
def has_significant_overlap(phrase: str, selected_phrases: List[str]) -> bool:
    """Check if more than 50% of the phrase is repeated in any selected phrase."""
    words = phrase.lower().split()
    if not words:
        return False
    for selected in selected_phrases:
        selected_words = selected.lower().split()
        common_words = set(words) & set(selected_words)
        if len(common_words) / len(words) > 0.5:
            return True
    return False

# --- New functions for balanced speaker selection ---
def get_speaker_distribution(voice_lines: List[VoiceLine]) -> Counter:
    """Calculate the distribution of speakers in the given voice lines."""
    return Counter(vl.VoiceType for vl in voice_lines)

def calculate_speaker_balance_score(voice_line: VoiceLine, 
                                    local_speaker_counts: Counter, 
                                    global_speaker_counts: Counter) -> float:
    """
    Calculate a balance score for a voice line based on how represented its 
    speaker is both locally (in the current phrase) and globally.
    
    Lower score means the speaker is more underrepresented (preferred for selection).
    Higher score means the speaker is more overrepresented (preferred for removal).
    """
    speaker = voice_line.VoiceType
    
    # Calculate local representation (within this phrase)
    local_total = sum(local_speaker_counts.values()) or 1  # Avoid division by zero
    local_proportion = local_speaker_counts[speaker] / local_total
    
    # Calculate global representation (across all selected phrases)
    global_total = sum(global_speaker_counts.values()) or 1  # Avoid division by zero
    global_proportion = global_speaker_counts[speaker] / global_total
    
    # Weight both factors (can be adjusted to prioritize local or global balance)
    # Higher weights for local_weight will prioritize diversity within phrases more
    local_weight = 0.7
    global_weight = 0.3
    
    return (local_proportion * local_weight) + (global_proportion * global_weight)

def select_balanced_voice_lines(potential_voice_lines: List[Tuple[VoiceLine, SentencePosition, Optional[str]]],
                               global_speaker_counts: Counter,
                               min_appearances: int,
                               target_appearances: int,
                               max_appearances: int,
                               current_phrase_obj: Phrase) -> List[VoiceLine]:
    """
    Select voice lines with a balanced distribution of speakers,
    considering both local (within phrase) and global (across all phrases) balance.
    
    Args:
        potential_voice_lines: List of tuples (voice_line, position, punctuation)
        global_speaker_counts: Counter tracking speaker distributions globally
        min/target/max_appearances: Configuration parameters for selection
        current_phrase_obj: The phrase object we're selecting for
        
    Returns:
        List of selected voice lines
    """
    selected_voice_lines = []
    selected_transcriptions = []
    
    # Track speaker distribution within this phrase
    local_speaker_counts = Counter()
    
    # First, include voice lines needed for different punctuation types if available
    punctuation_needed = {p: not current_phrase_obj.determining.punctuation_types.get(p, False) 
                          for p in [',', '.', '!', '?']}
    
    for punc in punctuation_needed:
        if not punctuation_needed[punc]:
            continue
        
        # Find voice lines with this punctuation
        punc_candidates = [item for item in potential_voice_lines 
                          if item[2] == punc and 
                          item[0].Transcription not in selected_transcriptions]
        
        if punc_candidates:
            # Sort by speaker balance score
            punc_candidates.sort(key=lambda item: calculate_speaker_balance_score(
                item[0], local_speaker_counts, global_speaker_counts))
            
            # Select the best candidate
            voice_line, position, _ = punc_candidates[0]
            selected_voice_lines.append(voice_line)
            selected_transcriptions.append(voice_line.Transcription)
            local_speaker_counts[voice_line.VoiceType] += 1
            current_phrase_obj.determining.punctuation_types[punc] = True
            
            # Update position counts
            position_key = position.name.lower()
            current_phrase_obj.determining.sentence_positions[position_key] += 1
    
    # Then, fill up to target_appearances with a diverse set of speakers
    remaining_slots = target_appearances - len(selected_voice_lines)
    
    # Filter out already selected voice lines
    remaining_candidates = [item for item in potential_voice_lines 
                           if item[0].Transcription not in selected_transcriptions]
    
    # Now select in rounds, each round picking the most underrepresented speaker
    while remaining_slots > 0 and remaining_candidates:
        # Sort by speaker balance score (prioritizing underrepresented speakers)
        remaining_candidates.sort(key=lambda item: calculate_speaker_balance_score(
            item[0], local_speaker_counts, global_speaker_counts))
        
        # Select the best candidate
        voice_line, position, punc = remaining_candidates[0]
        selected_voice_lines.append(voice_line)
        selected_transcriptions.append(voice_line.Transcription)
        local_speaker_counts[voice_line.VoiceType] += 1
        
        # Update position counts
        position_key = position.name.lower()
        current_phrase_obj.determining.sentence_positions[position_key] += 1
        
        # Update remaining candidates and slots
        remaining_candidates = [item for item in remaining_candidates 
                               if item[0].Transcription not in selected_transcriptions]
        remaining_slots -= 1
    
    # If we haven't reached minimum appearances, continue adding
    while len(selected_voice_lines) < min_appearances and remaining_candidates:
        # Take the next best candidate
        voice_line, position, punc = remaining_candidates[0]
        selected_voice_lines.append(voice_line)
        selected_transcriptions.append(voice_line.Transcription)
        remaining_candidates = remaining_candidates[1:]
        
        # Update counts
        local_speaker_counts[voice_line.VoiceType] += 1
        position_key = position.name.lower()
        current_phrase_obj.determining.sentence_positions[position_key] += 1
    
    # If we have more than max_appearances, trim the most overrepresented speakers
    if len(selected_voice_lines) > max_appearances:
        # Sort by overrepresentation (highest first), preserving punctuation lines
        # First, ensure we preserve at least one of each punctuation type
        preserved_punctuation_lines = set()
        for punc in [',', '.', '!', '?']:
            if current_phrase_obj.determining.punctuation_types.get(punc, False):
                # Find the first voice line with this punctuation
                for i, (vl, _, p) in enumerate(potential_voice_lines):
                    if p == punc and vl in selected_voice_lines:
                        preserved_punctuation_lines.add(vl)
                        break
        
        # Score voice lines for trimming (excluding preserved punctuation lines)
        trimmable_voice_lines = [vl for vl in selected_voice_lines 
                                if vl not in preserved_punctuation_lines]
        
        if len(trimmable_voice_lines) > (max_appearances - len(preserved_punctuation_lines)):
            # Score voice lines by how overrepresented their speakers are
            scored_voice_lines = [(vl, calculate_speaker_balance_score(vl, local_speaker_counts, global_speaker_counts)) 
                                 for vl in trimmable_voice_lines]
            
            # Sort by score (most overrepresented first)
            scored_voice_lines.sort(key=lambda x: x[1], reverse=True)
            
            # Keep only enough to stay under max_appearances
            keep_count = max_appearances - len(preserved_punctuation_lines)
            kept_trimmable = [vl for vl, _ in scored_voice_lines[:keep_count]]
            
            # Final selected set is preserved punctuation lines plus kept trimmable ones
            selected_voice_lines = list(preserved_punctuation_lines) + kept_trimmable
    
    # Update the speakers set for the phrase determination
    current_phrase_obj.determining.speakers = set(vl.VoiceType for vl in selected_voice_lines)
    current_phrase_obj.determining.speaker_counts = Counter(vl.VoiceType for vl in selected_voice_lines)
    
    return selected_voice_lines
  
# --- Main Training Data Selection Function ---
def select_training_data(voice_lines: VoiceLines, phrases_file: str,
                         min_appearances: int = MIN_APPEARANCES,
                         target_appearances: int = TARGET_APPEARANCES,
                         max_appearances: int = MAX_APPEARANCES) -> Tuple[SelectedTrainingData, Dict[str, str], List[VoiceLine]]:
    """
    Select training data based on the phrases and selection criteria.
    Returns:
        Tuple of (SelectedTrainingData, Dict of discarded phrases with reasons, List of discarded voice lines)
    """
    with open(phrases_file, 'r', encoding='utf-8') as f:
        phrase_texts = [line.strip() for line in f if line.strip()]
    phrases = {text: Phrase(text) for text in phrase_texts}
    discarded_phrases = {}
    discarded_voice_lines: List[VoiceLine] = []
    discarded_voice_line_ids: Set[str] = set()
    selected_training_data = SelectedTrainingData()
    all_voice_lines = voice_lines.lines.copy()
    random.shuffle(all_voice_lines)
    
    # Map phrases to voice lines that contain them using the new function.
    phrase_to_voice_lines = defaultdict(list)
    for voice_line in all_voice_lines:
        for phrase_text in phrase_texts:
            if voice_line_contains_phrase(voice_line, phrase_text):
                phrase_to_voice_lines[phrase_text].append(voice_line)
    
    selected_voice_line_ids: Set[str] = set()
    global_speaker_counts = Counter()  # Track global speaker distribution
    
    # Process each phrase
    for phrase_text, phrase_obj in phrases.items():
        available_voice_lines = phrase_to_voice_lines[phrase_text]
        
        # Check if we have enough voice lines for this phrase
        if len(available_voice_lines) < min_appearances:
            discarded_phrases[phrase_text] = f"Not enough voice lines (found {len(available_voice_lines)}, need {min_appearances})"
            # Add all available voice lines to discarded list
            for vl in available_voice_lines:
                if vl.InternalFileName not in discarded_voice_line_ids:
                    discarded_voice_lines.append(vl)
                    discarded_voice_line_ids.add(vl.InternalFileName)
            continue
        
        # Prepare potential voice lines for selection (filtering out already selected ones)
        potential_voice_lines = []
        selected_transcriptions = []
        
        for voice_line in available_voice_lines:
            if voice_line.InternalFileName in selected_voice_line_ids:
                continue
            
            # Check for significant overlap
            if has_significant_overlap(voice_line.Transcription, selected_transcriptions):
                continue
            
            # Get position and punctuation
            position = determine_sentence_position(voice_line.Transcription, phrase_text)
            punctuation = get_ending_punctuation(voice_line.Transcription, phrase_text)
            
            potential_voice_lines.append((voice_line, position, punctuation))
        
        # Use our new balanced selection function
        selected_lines = select_balanced_voice_lines(
            potential_voice_lines,
            global_speaker_counts,
            min_appearances,
            target_appearances,
            max_appearances,
            phrase_obj
        )
        
        # Store the selected voice lines
        phrase_obj.determining.selected_voice_lines = selected_lines
        
        # Mark selected voice lines as used
        for vl in selected_lines:
            selected_voice_line_ids.add(vl.InternalFileName)
        
        # Check if we have enough voice lines after selection
        if len(phrase_obj.determining.selected_voice_lines) < min_appearances:
            discarded_phrases[phrase_text] = f"Could not meet minimum appearances after filtering (found {len(phrase_obj.determining.selected_voice_lines)}, need {min_appearances})"
            for vl in phrase_obj.determining.selected_voice_lines:
                if vl.InternalFileName not in discarded_voice_line_ids:
                    discarded_voice_lines.append(vl)
                    discarded_voice_line_ids.add(vl.InternalFileName)
            continue
        
        # Check if we have at least two speakers
        if len(phrase_obj.determining.speakers) < 2:
            discarded_phrases[phrase_text] = "Only one speaker available"
            for vl in phrase_obj.determining.selected_voice_lines:
                if vl.InternalFileName not in discarded_voice_line_ids:
                    discarded_voice_lines.append(vl)
                    discarded_voice_line_ids.add(vl.InternalFileName)
            continue
        
        # Update global speaker counts with the selected voice lines
        for vl in phrase_obj.determining.selected_voice_lines:
            global_speaker_counts[vl.VoiceType] += 1
        
        # Store the final selection
        phrase_obj.voice_lines = phrase_obj.determining.selected_voice_lines
        selected_training_data.phrases.append(phrase_obj)
        
        # Add to overall voice lines collection
        for vl in phrase_obj.voice_lines:
            if vl not in selected_training_data.voice_lines:
                selected_training_data.voice_lines.append(vl)
    
    # Final stats output
    if selected_training_data.phrases:
        # Calculate speaker distribution across all selected data
        total_speaker_dist = Counter(vl.VoiceType for vl in selected_training_data.voice_lines)
        total_lines = len(selected_training_data.voice_lines)
        
        print("\nFinal speaker distribution:")
        for speaker, count in total_speaker_dist.most_common():
            print(f"  {speaker}: {count} lines ({count/total_lines*100:.1f}%)")
        
        # Calculate average number of unique speakers per phrase
        avg_speakers = sum(len(p.determining.speakers) for p in selected_training_data.phrases) / len(selected_training_data.phrases)
        print(f"\nAverage unique speakers per phrase: {avg_speakers:.2f}")
    
    return selected_training_data, discarded_phrases, discarded_voice_lines
  
def main():
    print_stage_header("Stage 2: Selecting Training Data")
    if not os.path.exists(WORDS_FILE):
        print(f"Error: {WORDS_FILE} not found")
        sys.exit(1)
    voice_lines = VoiceLines.load_from_yaml(INPUT_FILE)
    training_data, discarded_phrases, discarded_voice_lines = select_training_data(voice_lines, PHRASES_FILE)
    # Save selected training data in the new format: a YAML with count and lines.
    training_data.save_to_yaml(OUTPUT_TRAINING_DATA)
    # Write discarded voice lines in the same format as training data YAML.
    discarded_data = {
        "count": len(discarded_voice_lines),
        "lines": [vl.to_dict() for vl in discarded_voice_lines]
    }
    with open(OUTPUT_DISCARDED, 'w', encoding='utf-8') as f:
        yaml.dump(discarded_data, f, sort_keys=False, default_flow_style=False)
    print(f"We have {training_data.duration_minutes/60:.2f} hours of training data")
    print(f"Saved {len(discarded_voice_lines)} discarded voice lines to {OUTPUT_DISCARDED}")
    print(f"Stage 2 complete. Selected {len(training_data.phrases)} phrases with {len(training_data.voice_lines)} voice lines.")
  
if __name__ == "__main__":
    main()