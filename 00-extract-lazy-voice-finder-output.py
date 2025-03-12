import os
import shutil
import sys
import subprocess
import concurrent.futures

# Make sure to use raw strings for Windows paths to avoid escape issues
LAZY_VOICE_FINDER_OUTPUT_DIR = r"C:\Users\mikas\Downloads\VoiceExtra\LazyVoiceFinder\Export\sound\voice"
VOICE_FILE_DIR = r"C:\Users\mikas\Downloads\VoiceExtra\VoiceFiles"  # destination directory for WAV files

def move_fuz_files(root_dir):
    # Walk the directory tree from bottom-up so that subdirectories are processed before their parents
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        # Skip processing the root directory itself
        if os.path.abspath(dirpath) == os.path.abspath(root_dir):
            continue
        for filename in filenames:
            if filename.lower().endswith('.fuz') or filename.lower().endswith('.xwm'):
                file_path = os.path.join(dirpath, filename)
                rel_path_parts = os.path.relpath(dirpath, root_dir).split(os.sep)
                # Ignore the first directory in the path (plugin name)
                rel_path = "_".join(rel_path_parts[1:]) if len(rel_path_parts) > 1 else ""
                new_name = filename if not rel_path else rel_path + '_' + filename
                new_path = os.path.join(root_dir, new_name)
                counter = 1
                base_new_name, ext = os.path.splitext(new_name)
                while os.path.exists(new_path):
                    new_name = f"{base_new_name}_{counter}{ext}"
                    new_path = os.path.join(root_dir, new_name)
                    counter += 1
                shutil.move(file_path, new_path)
                print(f"Moved: {file_path} -> {new_path}")
        # Remove directory regardless of whether it is empty
        shutil.rmtree(dirpath, ignore_errors=True)
        print(f"Removed directory: {dirpath}")

def convert_to_xwm():
    os.system(f'pwsh -ExecutionPolicy Bypass -File scripts\\convert_fuz_to_xwm.ps1 -CustomDir "{LAZY_VOICE_FINDER_OUTPUT_DIR}"')

# def cleanup_export_dir():
#    os.rmdir(LAZY_VOICE_FINDER_OUTPUT_DIR)

def convert_to_wav():
    num_workers = 4

    # Create the destination directory if it doesn't exist.
    if not os.path.exists(VOICE_FILE_DIR):
        os.makedirs(VOICE_FILE_DIR)
    
    # List all .xwm files in the output directory
    xwm_files = [
        os.path.join(LAZY_VOICE_FINDER_OUTPUT_DIR, f)
        for f in os.listdir(LAZY_VOICE_FINDER_OUTPUT_DIR)
        if f.lower().endswith('.xwm')
    ]
    
    def convert_file(xwm_file):
        base_name = os.path.splitext(os.path.basename(xwm_file))[0]
        output_wav = os.path.join(VOICE_FILE_DIR, base_name + ".wav")
        # Only convert if the WAV file does not already exist.
        if os.path.exists(output_wav):
            return

        # Build the ffmpeg command
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-i", xwm_file,
            "-ar", "16000",
            output_wav
        ]
        try:
            print(output_wav)
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error converting {xwm_file}: {e}")

    # Run conversions in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(convert_file, xwm_files)

if __name__ == "__main__":
    # move_fuz_files(LAZY_VOICE_FINDER_OUTPUT_DIR)
    # convert_to_xwm()
    convert_to_wav()
    # cleanup_export_dir()
