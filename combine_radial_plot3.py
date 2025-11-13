# Filename: plot_rotated_point_cloud_v3.py
# Place this in your main 'OpenRadar' directory (with azimuth.py)
# and run it from there.

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
    input_file: str = '' # This will be set in the loop
    
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
            print(f"  WARNING: Skipping file {input_file}. Expected 2D array, got {raw_data.ndim}D.")
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
    print("Starting point cloud generation and rotation...")
    
    # 1. Define datasets, their rotation, and their color
    datasets_to_process = {
        # 'Folder Name': (Rotation Angle, Color)
        '0_degree': (0.0, 'blue'),
        '15_degree_positive': (-15.0, 'green'),
        '15_degree_negative': (15.0, 'red'),
        '30_degree_positive': (-30.0, 'cyan'),
        '30_degree_negative': (30.0, 'magenta')
    }
    
    config = RadarConfig()
    processor = AzimuthProcessor(config)
    
    # Store points in a dictionary by dataset
    all_points_by_dataset = {name: [] for name in datasets_to_process}
    
    # Create axes for finding coordinates from heatmap indices
    angle_axis_deg = np.linspace(-config.angle_range, config.angle_range, (2 * config.angle_range // config.angle_res) + 1)
    range_axis_m = np.linspace(0, config.range_resolution * config.bins_processed, config.bins_processed)
    
    # 2. Loop through each dataset folder
    for folder, (rotation_angle_deg, color) in datasets_to_process.items():
        if not os.path.isdir(f'./{folder}'):
            print(f"  WARNING: Directory ./{folder}/ not found. Skipping.")
            continue
            
        print(f"\nProcessing folder: ./{folder}/ (Rotation: {rotation_angle_deg} degrees)")
        
        file_paths = glob.glob(f'./{folder}/*.npy')
        if not file_paths:
            print(f"  WARNING: No .npy files found in ./{folder}/. Skipping.")
            continue
            
        # 3. Loop through each file in the folder
        for file_path in file_paths:
            try:
                organized_frames = processor.load_and_organize_data(file_path)
            except Exception as e:
                print(f"  Error loading {file_path}: {e}. Skipping file.")
                continue

            # 4. Loop through each valid frame from the file
            for frame in organized_frames:
                # 5. Process frame (clutter removal + Gaussian)
                heatmap = processor.process_frame(frame)
                
                # 6. Find the single highest point
                az_idx, range_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                
                # 7. Convert indices to physical units
                detected_angle_deg = angle_axis_deg[az_idx]
                detected_range_m = range_axis_m[range_idx]
                
                # 8. Apply rotational transform
                rotated_angle_deg = detected_angle_deg + rotation_angle_deg
                
                # 9. Convert polar (r, theta) to Cartesian (x, y)
                rotated_angle_rad = np.deg2rad(rotated_angle_deg)
                x = detected_range_m * np.sin(rotated_angle_rad)
                y = detected_range_m * np.cos(rotated_angle_rad)
                
                # 10. Store point in its colored group
                all_points_by_dataset[folder].append((x, y))

    print("\nProcessing complete. All points collected.")
    
    # 11. Create the final scatter plot with colors
    print("Generating plot...")
    plt.figure(figsize=(12, 12))
    
    for folder, points in all_points_by_dataset.items():
        if not points:
            print(f"No points found for {folder}")
            continue
            
        color = datasets_to_process[folder][1]
        label = folder
        
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        plt.scatter(x_coords, y_coords, alpha=0.05, s=15, color=color, label=label)

    # Format the plot
    plt.title('Rotated and Aligned Point Cloud (Color-Coded)')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True, linestyle='--')
    plt.axis('equal') 
    
    # --- ADD THE LEGEND ---
    legend = plt.legend(loc='upper right')
    
    # --- THIS IS THE FIX ---
    # Change legend.legendHandles to legend.legend_handles
    for handle in legend.legend_handles:
        handle.set_alpha(1) # Make legend markers opaque
    
    output_filename = 'rotated_point_cloud_colored_plot.png'
    plt.savefig(output_filename)
    print(f"Successfully saved plot to '{output_filename}'")
    plt.show()

if __name__ == "__main__":
    main()