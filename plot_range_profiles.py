# Filename: plot_range_profiles.py
# Place this in your main 'OpenRadar' directory and run it.
# This script plots the 1D range profile at the peak angle for all frames.

import sys
sys.path.append('.') # Add current directory to path to find 'mmwave'

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import glob
import os

# --- Imports from your project ---
# We use the 'dsp' module from your local 'mmwave' folder
from mmwave import dsp 
# We use the Gaussian filter from scipy, as seen in your test script
from scipy.ndimage import gaussian_filter 

# -----------------------------------------------------------------
# Configuration class from test_data_processing_gaussian.py
# -----------------------------------------------------------------
@dataclass
class RadarConfig:
    """Configuration parameters for radar processing"""
    # File parameters
    # --- We will use only this one file ---
    input_file: str = './0_degree/0_degree_positive1_2025-10-16_15-25-11.npy'
    
    # Radar parameters
    num_chirps: int = 182 * 3
    num_rx: int = 4
    num_samples: int = 256
    num_tx: int = 3
    
    # Processing parameters
    range_resolution: float = 0.0488
    angle_range: int = 90
    angle_res: int = 1
    virt_ant: int = 12
    bins_processed: int = 256
    
    # Visualization and filters from test_data_processing_gaussian.py
    noise_filter_size: int = 3
    dynamic_range_db: float = 40

# -----------------------------------------------------------------
# Processor class based on test_data_processing_gaussian.py
# -----------------------------------------------------------------
class AzimuthProcessor:
    def __init__(self, config):
        self.config = config
        self.steering_vec = dsp.gen_steering_vec(
            config.angle_range, 
            config.angle_res,
            config.virt_ant
        )[1] 
    
    def load_and_organize_data(self, input_file):
        """
        Load and organize a specific file, correctly handling multi-frame arrays.
        """
        self.config.input_file = input_file
        raw_data = np.load(self.config.input_file, allow_pickle=True)
        
        if raw_data.ndim != 2:
            print(f"  ERROR: Skipping file {input_file}. Expected 2D array, got {raw_data.ndim}D.")
            return []

        num_frames = raw_data.shape[0]
        print(f"  Loading {num_frames} frames from {self.config.input_file}...")

        organized_data_list = []
        for i in range(num_frames):
            frame_data = raw_data[i] # Get the i-th row (one frame)
            try:
                organized_data_list.append(self._organize_iq(
                    frame_data,
                    num_chirps=self.config.num_chirps,
                    num_rx=self.config.num_rx,
                    num_samples=self.config.num_samples
                ))
            except ValueError as e:
                print(f"    Skipping frame {i+1}/{num_frames} due to error: {e}")
        
        return organized_data_list
    
    def _organize_iq(self, data, num_chirps, num_rx, num_samples):
        """Reshape raw ADC data into complex format"""
        expected_length = num_chirps * num_rx * num_samples * 2
        
        if data.size != expected_length:
            raise ValueError(f"Data length {data.size} != expected {expected_length}")
        
        data_reshaped = data.reshape(num_chirps, num_rx, num_samples, 2)
        return data_reshaped[..., 0] + 1j * data_reshaped[..., 1]
    
    def process_frame(self, frame):
        """
        Full processing pipeline for one frame, with
        clutter removal AND Gaussian filter.
        """
        # 1. Range FFT
        radar_cube = dsp.range_processing(frame)
        
        # 2. Static Clutter Removal (as requested)
        radar_cube = dsp.clutter_removal(radar_cube)
        
        # 3. Virtual antenna arrangement
        radar_cube = np.concatenate(
            [radar_cube[i::self.config.num_tx, ...] 
             for i in range(self.config.num_tx)],
            axis=1
        )
        # 4. Capon beamforming
        range_azimuth = np.zeros((self.steering_vec.shape[0], self.config.bins_processed))
        for i in range(self.config.bins_processed):
            range_azimuth[:, i], _ = dsp.aoa_capon(
                radar_cube[:, :, i].T, 
                self.steering_vec, 
                magnitude=True
            )
        
        # 5. Apply Gaussian Filter (as requested)
        if self.config.noise_filter_size > 0:
            range_azimuth = gaussian_filter(range_azimuth, sigma=self.config.noise_filter_size)
            
        # 6. Generate heatmap (log magnitude)
        return self._generate_heatmap(range_azimuth)
    
    def _generate_heatmap(self, range_azimuth):
        """Create visualization-ready heatmap (power in dB)"""
        heatmap = 10 * np.log10(range_azimuth + 1e-10)
        max_power = np.max(heatmap)
        heatmap = np.clip(heatmap, max_power-self.config.dynamic_range_db, max_power)
        return heatmap

# -----------------------------------------------------------------
# Main function to run the analysis
# -----------------------------------------------------------------
def main():
    print("Starting 1D Range Profile plot...")
    
    config = RadarConfig() # Config now has the file path hard-coded
    
    if not os.path.exists(config.input_file):
        print(f"ERROR: File not found: {config.input_file}")
        print("Please check the 'input_file' variable in RadarConfig.")
        return
        
    processor = AzimuthProcessor(config)
    
    # --- NEW: List to store all 1D range profiles ---
    all_range_profiles = []
    
    # Create x-axis for plotting
    range_axis_m = np.linspace(0, config.range_resolution * config.bins_processed, config.bins_processed)
    
    # --- Load and process the single file ---
    try:
        organized_frames = processor.load_and_organize_data(config.input_file)
    except Exception as e:
        print(f"  Error loading {config.input_file}: {e}. Exiting.")
        return

    if not organized_frames:
        print("No valid frames were loaded.")
        return

    print(f"Processing {len(organized_frames)} frames...")
    
    # --- Loop through each frame ---
    for frame in organized_frames:
        # 1. Process frame (clutter removal + Gaussian)
        heatmap = processor.process_frame(frame)
        
        # 2. Find the (angle index) of the single highest point
        az_idx, range_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        
        # 3. --- NEW: Extract the 1D Range Profile ---
        #    This is the full 1D slice at the detected angle
        range_profile_at_peak_angle = heatmap[az_idx, :]
        
        all_range_profiles.append(range_profile_at_peak_angle)

    print(f"\nProcessing complete. Collected {len(all_range_profiles)} profiles.")
    
    # 4. --- Create the final 1D plot ---
    print("Generating plot...")
    plt.figure(figsize=(12, 8))
    
    # Plot all frames as "shades of lines"
    for profile in all_range_profiles:
        plt.plot(range_axis_m, profile, color='blue', alpha=0.05)
        
    # Calculate and plot the average profile
    mean_profile = np.mean(all_range_profiles, axis=0)
    plt.plot(range_axis_m, mean_profile, color='red', linewidth=2, label='Average Profile')

    # Format the plot
    plt.title(f'Range Profile at Peak Angle (All Frames)\nFile: {config.input_file}')
    plt.xlabel('Range (meters)')
    plt.ylabel('Power (dB)')
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.xlim(0, config.range_resolution * config.bins_processed) # Full range
    
    # Try to zoom in on the action
    try:
        peak_range_index = np.argmax(mean_profile)
        peak_range_m = range_axis_m[peak_range_index]
        plt.xlim(peak_range_m - 2, peak_range_m + 2) # Zoom to +/- 2m of peak
    except Exception as e:
        print(f"Could not auto-zoom: {e}")

    output_filename = 'range_profile_distribution.png'
    plt.savefig(output_filename)
    print(f"Successfully saved plot to '{output_filename}'")
    plt.show() # Also show the plot

if __name__ == "__main__":
    main()
