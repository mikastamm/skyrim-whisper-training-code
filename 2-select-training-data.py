#!/usr/bin/env python
import yaml
import random
import os
from typing import List, Dict, Any, Set, Optional, Tuple
from collections import defaultdict
import sys
from enum import Enum, auto
import re

from Pipeline import VoiceLine, VoiceLines, print_stage_header, VOICE_FILE_DIR, PHRASES_FILE

class PhraseDetermination:
    """Intermediate variables for the phrase selection process."""
    def __init__(self):
        self.selected_voice_lines: List[VoiceLine] = []
        self.speakers: Set[str] = set()  # Track unique speakers
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
        """Save the selected training data to a YAML file."""
        data = {
            "phrases": [phrase.to_dict() for phrase in self.phrases],
            "voice_lines": [vl.to_dict() for vl in self.voice_lines],
            "duration_minutes": self.duration_minutes,
            "total_voice_lines": len(self.voice_lines)
        }
        
        with open(file_path, 'w', encoding='utf-8') as file:
            yaml.dump(data, file, sort_keys=False, default_flow_style=False)
        
        print(f"Saved selected training data with {len(self.phrases)} phrases and {len(self.voice_lines)} voice lines to {file_path}")
        print(f"Total duration: {self.duration_minutes:.2f} minutes")

def determine_sentence_position(transcription: str, phrase: str) -> SentencePosition:
    """Determine the position of the phrase within the transcription."""
    # Strip punctuation for better matching
    clean_transcription = re.sub(r'[^\w\s]', '', transcription.lower())
    clean_phrase = re.sub(r'[^\w\s]', '', phrase.lower())
    
    # Find where the phrase appears in the transcription
    phrase_index = clean_transcription.find(clean_phrase)
    
    if phrase_index == -1:
        # Fallback if exact match not found (shouldn't happen if filtering worked correctly)
        return SentencePosition.MIDDLE
    
    total_length = len(clean_transcription)
    phrase_end_index = phrase_index + len(clean_phrase)
    
    # Determine position based on relative location
    if phrase_index < total_length * 0.25:
        return SentencePosition.START
    elif phrase_end_index > total_length * 0.75:
        return SentencePosition.END
    else:
        return SentencePosition.MIDDLE

def get_ending_punctuation(transcription: str, phrase: str) -> Optional[str]:
    """Get the punctuation mark following the phrase in the transcription."""
    # Simple method to find punctuation after the phrase
    clean_phrase = phrase.lower().strip()
    transcription_lower = transcription.lower()
    
    # Find the phrase in the transcription
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

def select_training_data(voice_lines: VoiceLines, phrases_file: str, 
                         min_appearances: int = 3, target_appearances: int = 5, 
                         max_appearances: int = 8) -> Tuple[SelectedTrainingData, Dict[str, str]]:
    """
    Select training data based on the phrases and selection criteria.
    
    Returns:
        Tuple of (SelectedTrainingData, Dict of discarded phrases with reasons)
    """
    # Read phrases from file
    with open(phrases_file, 'r', encoding='utf-8') as f:
        phrase_texts = [line.strip() for line in f if line.strip()]
    
    # Initialize data structures
    phrases = {text: Phrase(text) for text in phrase_texts}
    discarded_phrases = {}
    selected_training_data = SelectedTrainingData()
    
    # Randomize voice lines order
    all_voice_lines = voice_lines.lines.copy()
    random.shuffle(all_voice_lines)
    
    # Map phrases to voice lines that contain them
    phrase_to_voice_lines = defaultdict(list)
    for voice_line in all_voice_lines:
        transcription = voice_line.Transcription.lower()
        for phrase_text in phrase_texts:
            # Check if phrase appears in the transcription as a whole word/phrase
            if re.search(r'\b' + re.escape(phrase_text.lower()) + r'\b', transcription):
                phrase_to_voice_lines[phrase_text].append(voice_line)
    
    # Track already selected voice lines to ensure each only appears once
    selected_voice_line_ids = set()
    
    # First pass: Try to achieve minimum criteria for each phrase
    for phrase_text, phrase_obj in phrases.items():
        available_voice_lines = phrase_to_voice_lines[phrase_text]
        
        # Check if we have enough voice lines for this phrase
        if len(available_voice_lines) < min_appearances:
            discarded_phrases[phrase_text] = f"Not enough voice lines (found {len(available_voice_lines)}, need {min_appearances})"
            continue
        
        # Track which transcription segments we've already selected to avoid >50% overlap
        selected_transcriptions = []
        
        # Try to get diverse punctuation, speakers, and positions
        for voice_line in available_voice_lines:
            # Skip if this voice line is already selected
            if voice_line.InternalFileName in selected_voice_line_ids:
                continue
                
            # Skip if there's significant overlap with already selected transcriptions
            if has_significant_overlap(voice_line.Transcription, selected_transcriptions):
                continue
            
            # Determine sentence position and ending punctuation
            position = determine_sentence_position(voice_line.Transcription, phrase_text)
            punctuation = get_ending_punctuation(voice_line.Transcription, phrase_text)
            
            # Check if this position is preferred based on our current selections
            position_key = position.name.lower()
            
            # Add to selections if we're under the target or if this helps balance positions
            if (len(phrase_obj.determining.selected_voice_lines) < target_appearances and
                (not punctuation or not phrase_obj.determining.punctuation_types.get(punctuation, False))):
                
                phrase_obj.determining.selected_voice_lines.append(voice_line)
                phrase_obj.determining.speakers.add(voice_line.VoiceType)
                selected_voice_line_ids.add(voice_line.InternalFileName)
                selected_transcriptions.append(voice_line.Transcription)
                
                # Track punctuation and position
                if punctuation in phrase_obj.determining.punctuation_types:
                    phrase_obj.determining.punctuation_types[punctuation] = True
                phrase_obj.determining.sentence_positions[position_key] += 1
                
                # Stop if we've reached the max appearances
                if len(phrase_obj.determining.selected_voice_lines) >= max_appearances:
                    break
        
        # Check if we have at least the minimum number of appearances
        if len(phrase_obj.determining.selected_voice_lines) < min_appearances:
            discarded_phrases[phrase_text] = f"Could not meet minimum appearances after filtering (found {len(phrase_obj.determining.selected_voice_lines)}, need {min_appearances})"
            continue
            
        # Check if we have more than one speaker
        if len(phrase_obj.determining.speakers) < 2:
            discarded_phrases[phrase_text] = "Only one speaker available"
            continue
            
        # Phrase passes all criteria, add to final selection
        phrase_obj.voice_lines = phrase_obj.determining.selected_voice_lines
        selected_training_data.phrases.append(phrase_obj)
        
        # Also add to the global voice_lines list
        for vl in phrase_obj.voice_lines:
            if vl not in selected_training_data.voice_lines:
                selected_training_data.voice_lines.append(vl)
    
    return selected_training_data, discarded_phrases

def main():
    print_stage_header("Stage 2: Selecting Training Data")
    
    # Check if words.txt exists
    if not os.path.exists("words.txt"):
        print("Error: words.txt not found")
        sys.exit(1)
    
    # Load filtered voice lines
    voice_lines = VoiceLines.load_from_yaml("1-filtered.yaml")
    
    # Select training data
    training_data, discarded_phrases = select_training_data(voice_lines, PHRASES_FILE)
    
    # Save selected training data
    training_data.save_to_yaml("2-training-data.yaml")
    
    # Save discarded phrases
    with open("2-discarded.yaml", 'w', encoding='utf-8') as f:
        yaml.dump({"discarded_phrases": discarded_phrases}, f, sort_keys=False, default_flow_style=False)
    
    print(f"We have {training_data.duration_minutes/60} hours of training data")
    print(f"Saved {len(discarded_phrases)} discarded phrases to 2-discarded.yaml")
    print(f"Stage 2 complete. Selected {len(training_data.phrases)} phrases with {len(training_data.voice_lines)} voice lines.")

if __name__ == "__main__":
    main()