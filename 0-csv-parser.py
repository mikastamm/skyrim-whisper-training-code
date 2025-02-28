#!/usr/bin/env python
"""
Stage 0: CSV Parsing
Parses the lazy-voice-finder-output.csv file and creates VoiceLine objects.
"""
import os
import csv
import sys
from Pipeline import VoiceLine, VoiceLines, print_stage_header, Fore, Style

# Configuration
INPUT_CSV_FILE = "./lazy-voice-finder-output.csv"  # Input CSV file
OUTPUT_YAML_FILE = "./0-parsed.yaml"  # Output YAML file

def main():
    """Main function to parse CSV and create VoiceLine objects."""
    print_stage_header("Stage 0: CSV Parsing")
    
    # Check if input file exists
    if not os.path.exists(INPUT_CSV_FILE):
        print(f"{Fore.RED}Error: Input CSV file {INPUT_CSV_FILE} not found.")
        sys.exit(1)
    
    voice_lines = VoiceLines()
    
    # Read CSV file
    try:
        with open(INPUT_CSV_FILE, 'r', encoding='utf-8-sig') as csv_file:
            # Create CSV reader with proper dialect
            csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')
            
            # Read header
            headers = next(csv_reader)
            expected_headers = ["State", "Plugin", "File Name", "Voice Type", 
                               "Dialogue 1 - English", "Dialogue 2 - English"]
            
            # Verify headers
            if not all(h in headers for h in expected_headers):
                print(f"{Fore.RED}Error: CSV headers don't match expected format.")
                print(f"{Fore.RED}Expected: {expected_headers}")
                print(f"{Fore.RED}Found: {headers}")
                sys.exit(1)
            
            # Get column indices
            state_idx = headers.index("State")
            plugin_idx = headers.index("Plugin")
            file_name_idx = headers.index("File Name")
            voice_type_idx = headers.index("Voice Type")
            dialogue1_idx = headers.index("Dialogue 1 - English")
            dialogue2_idx = headers.index("Dialogue 2 - English")
            
            # Process rows
            row_count = 0
            for row in csv_reader:
                row_count += 1
                
                # Skip rows with insufficient data
                if len(row) < len(headers):
                    print(f"{Style.DIM}Skipping row {row_count}: insufficient columns")
                    continue
                
                # Extract data
                state = row[state_idx]
                plugin = row[plugin_idx]
                internal_file_name = row[file_name_idx]
                voice_type = row[voice_type_idx]
                
                # Get transcription (prefer Dialogue 1, fallback to Dialogue 2)
                transcription = row[dialogue1_idx]
                if not transcription and len(row) > dialogue2_idx:
                    transcription = row[dialogue2_idx]
                
                # Skip rows without transcription
                if not transcription:
                    print(f"{Style.DIM}Skipping row {row_count}: no transcription")
                    continue
                
                # Create file name in required format (VoiceType_InternalFileName.wav)
                # Extract original extension
                orig_name_parts = internal_file_name.split('.')
                base_name = '.'.join(orig_name_parts[:-1]) if len(orig_name_parts) > 1 else internal_file_name
                file_name = f"{voice_type}_{base_name}.wav"
                
                # Create VoiceLine
                voice_line = VoiceLine(
                    internal_file_name=internal_file_name,
                    transcription=transcription,
                    voice_type=voice_type,
                    plugin=plugin,
                    state=state,
                    file_name=file_name
                )
                
                voice_lines.add(voice_line)
    
    except Exception as e:
        print(f"{Fore.RED}Error parsing CSV: {str(e)}")
        sys.exit(1)
    
    # Print summary
    print(f"{Fore.GREEN}Successfully parsed {len(voice_lines)} voice lines from CSV.")
    
    # Save to YAML
    voice_lines.save_to_yaml(OUTPUT_YAML_FILE)
    voice_lines.summarize()

if __name__ == "__main__":
    main()
