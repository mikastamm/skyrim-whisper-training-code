import yaml
import os
import colorama
from colorama import Fore, Style
from typing import List, Any, Optional, Dict
import sys

# Initialize colorama for colored console output
colorama.init(autoreset=True)

# continains the wav files
VOICE_FILE_DIR=os.path.join("..", "audio_final")
FILTERED_VOICE_FILE_DIR=os.path.join("..", "filtered_audio")
PHRASES_FILE="./words.txt"

class VoiceLine:
    """
    Represents a single voice line transcription entry from the dataset.
    """
    def __init__(self, internal_file_name: str = "", transcription: str = "", 
                 voice_type: str = "", plugin: str = "", state: str = "", 
                 file_name: str = "", duration_ms: int = -1, stage2_data: Any = None):
        self.InternalFileName: str = internal_file_name
        self.Transcription: str = transcription
        self.VoiceType: str = voice_type
        self.Plugin: str = plugin
        self.State: str = state
        self.FileName: str = file_name
        self.DurationMs: int = duration_ms
        self.Stage2Data: Any = stage2_data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the VoiceLine to a dictionary for YAML serialization."""
        return {
            "InternalFileName": self.InternalFileName,
            "Transcription": self.Transcription,
            "VoiceType": self.VoiceType,
            "Plugin": self.Plugin,
            "State": self.State,
            "FileName": self.FileName,
            "DurationMs": self.DurationMs,
            "Stage2Data": self.Stage2Data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VoiceLine':
        """Create a VoiceLine instance from a dictionary."""
        return cls(
            internal_file_name=data.get("InternalFileName", ""),
            transcription=data.get("Transcription", ""),
            voice_type=data.get("VoiceType", ""),
            plugin=data.get("Plugin", ""),
            state=data.get("State", ""),
            file_name=data.get("FileName", ""),
            duration_ms=data.get("DurationMs", -1),
            stage2_data=data.get("Stage2Data", None)
        )


class VoiceLines:
    """
    Container for a collection of VoiceLine objects. Supports loading/saving to YAML
    and tracking filtering operations.
    """
    def __init__(self, lines: Optional[List[VoiceLine]] = None):
        self.lines: List[VoiceLine] = lines or []
        self.original_count: int = len(self.lines)
    
    def add(self, line: VoiceLine) -> None:
        """Add a VoiceLine to the collection."""
        self.lines.append(line)
        self.original_count = len(self.lines)
    
    def save_to_yaml(self, file_path: str) -> None:
        """Save the collection to a YAML file."""
        data = {
            "count": len(self.lines),
            "lines": [line.to_dict() for line in self.lines]
        }
        
        with open(file_path, 'w', encoding='utf-8') as file:
            yaml.dump(data, file, sort_keys=False, default_flow_style=False)
        
        print(f"{Fore.GREEN}Saved {len(self.lines)} voice lines to {file_path}")
    
    @classmethod
    def load_from_yaml(cls, file_path: str) -> 'VoiceLines':
        """Load a collection from a YAML file."""
        if not os.path.exists(file_path):
            print(f"{Fore.RED}Error: File {file_path} not found.")
            sys.exit(1)
            
        with open(file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        
        if not data or "lines" not in data:
            print(f"{Fore.RED}Error: Invalid YAML format in {file_path}")
            sys.exit(1)
            
        lines = [VoiceLine.from_dict(line_data) for line_data in data["lines"]]
        result = cls(lines)
        result.original_count = data.get("count", len(lines))
        
        print(f"{Fore.GREEN}Loaded {len(lines)} voice lines from {file_path}")
        return result
    
    def filter(self, filter_name: str, predicate) -> 'VoiceLines':
        """
        Filter the collection based on a predicate function.
        Returns a new VoiceLines instance containing only the items that passed the filter.
        """
        initial_count = len(self.lines)
        filtered_lines = [line for line in self.lines if predicate(line)]
        filtered_count = initial_count - len(filtered_lines)
        
        result = VoiceLines(filtered_lines)
        result.original_count = self.original_count
        
        if filtered_count > 0:
            print(f"{Fore.YELLOW}[{filter_name}] Removed {filtered_count} voice lines "
                  f"({filtered_count / initial_count:.2%})")
        else:
            print(f"{Style.DIM}[{filter_name}] No voice lines removed")
            
        return result
    
    def __len__(self) -> int:
        return len(self.lines)
    
    def summarize(self) -> None:
        """Print a summary of the current collection state."""
        removed = self.original_count - len(self.lines)
        if removed > 0:
            print(f"\n{Fore.YELLOW}===== Summary =====")
            print(f"{Fore.YELLOW}Original count: {self.original_count}")
            print(f"{Fore.YELLOW}Current count: {len(self.lines)}")
            print(f"{Fore.YELLOW}Total removed: {removed} ({removed / self.original_count:.2%})")
        else:
            print(f"\n{Fore.GREEN}No voice lines were removed.")


def print_stage_header(stage_name: str) -> None:
    """Print a formatted header for a pipeline stage."""
    print(f"\n{Fore.CYAN}{'=' * 20}")
    print(f"{Fore.CYAN}{stage_name}")
    print(f"{Fore.CYAN}{'=' * 20}\n")
