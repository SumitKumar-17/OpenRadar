import os
import glob
import subprocess
import numpy as np

# --- Configuration ---

# Base directories
BASE_DATASET_DIR = 'dataset'
BASE_VIDEO_DIR = 'videos'
OUTPUT_GIF_DIR = 'output_gifs'

# Mapping of dataset subfolders to video subfolders
# (Based on your 'ls' output)
DIRECTORY_MAP = {
    '0_degree': '0_degree',
    '15_degree_negative': 'minus_15_degree',
    '15_degree_positive': '15_degree',
    '30_degree_negative': 'minus_30_degree',
    '30_degree_positive': '30_degree'
}

# Path to the processing script
PROCESSING_SCRIPT = 'azimuth_gif_creation.py'

# -----------------------

def process_all_files():
    """
    Finds all matching .npy and .mp4 files and runs the processing script.
    """
    print(f"Starting batch processing...")
    os.makedirs(OUTPUT_GIF_DIR, exist_ok=True)
    
    total_processed = 0
    total_skipped = 0
    
    # Iterate over the directory mapping
    for dataset_subdir, video_subdir in DIRECTORY_MAP.items():
        print(f"\n--- Processing Directory: {dataset_subdir} ---")
        
        dataset_path = os.path.join(BASE_DATASET_DIR, dataset_subdir)
        video_path = os.path.join(BASE_VIDEO_DIR, video_subdir)
        
        # Find all .npy files in the current dataset directory
        npy_files = glob.glob(os.path.join(dataset_path, '*.npy'))
        
        if not npy_files:
            print(f"No .npy files found in {dataset_path}")
            continue
            
        for npy_path in npy_files:
            # Get the base filename without the extension
            # e.g., 'minus_15_degree_1'
            base_filename = os.path.splitext(os.path.basename(npy_path))[0]
            
            # Construct the expected paths
            mp4_path = os.path.join(video_path, f"{base_filename}.mp4")
            output_gif_path = os.path.join(OUTPUT_GIF_DIR, f"{base_filename}.gif")
            
            print(f"\nProcessing: {base_filename}")
            
            # --- Validations ---
            if not os.path.exists(mp4_path):
                print(f"  [SKIP] Video file not found at: {mp4_path}")
                total_skipped += 1
                continue
                
            if os.path.exists(output_gif_path):
                print(f"  [SKIP] Output GIF already exists: {output_gif_path}")
                total_skipped += 1
                continue
                
            # --- Get Radar Frame Count ---
            try:
                data = np.load(npy_path, allow_pickle=True)
                radar_total_frames = len(data)
                del data # Free memory
                if radar_total_frames == 0:
                    raise ValueError("Radar file is empty.")
                radar_start = 0
                radar_end = radar_total_frames - 1
            except Exception as e:
                print(f"  [FAIL] Could not load radar file {npy_path}: {e}")
                total_skipped += 1
                continue
            
            print(f"  Radar file: {npy_path} ({radar_total_frames} frames)")
            print(f"  Video file: {mp4_path}")
            print(f"  Output file: {output_gif_path}")
            
            # --- Build and Run Command ---
            command = [
                'python',
                PROCESSING_SCRIPT,
                '--input_file', npy_path,
                '--video_file', mp4_path,
                '--radar_start', str(radar_start),
                '--radar_end', str(radar_end),
                '--output_gif', output_gif_path
            ]
            
            try:
                print("  Running processing script...")
                subprocess.run(command, check=True)
                print(f"  [SUCCESS] Created {output_gif_path}")
                total_processed += 1
            except subprocess.CalledProcessError as e:
                print(f"  [FAIL] Script failed for {base_filename}: {e}")
                total_skipped += 1
            except KeyboardInterrupt:
                print("\nBatch processing interrupted by user.")
                return

    print("\n" + "="*50)
    print("BATCH PROCESSING COMPLETE")
    print(f"  Successfully processed: {total_processed}")
    print(f"  Failed / Skipped:     {total_skipped}")
    print(f"  Output directory:     {OUTPUT_GIF_DIR}")
    print("="*50)

if __name__ == "__main__":
    process_all_files()