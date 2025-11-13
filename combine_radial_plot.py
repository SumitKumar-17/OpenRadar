# Filename: generate_radial_plot.py
# Run this script from your main 'OpenRadar' directory.

import numpy as np
import matplotlib.pyplot as plt
from mmwave import dsp  # This will import from your local 'mmwave' package
from dataclasses import dataclass
import glob
import os

# -----------------------------------------------------------------
# This RadarConfig class is copied from your azimuth.py
# -----------------------------------------------------------------
@dataclass
class RadarConfig:
    """Configuration parameters for radar processing"""
    # --- MODIFIABLE ---
    # Change this to the file you want to plot
    input_file: str = './0_degree/0_degree_positive1_2025-10-16_15-25-11.npy'
    
    # Radar parameters
    num_chirps: int = 182 * 3  # 546
    num_rx: int = 4
    num_samples: int = 256
    num_tx: int = 3
    
    # Processing parameters
    range_resolution: float = 0.0488  # meters
    angle_range: int = 90  # degrees
    angle_res: int = 1  # degrees
    virt_ant: int = 12
    bins_processed: int = 256
    
    # Visualization
    cmap: str = 'viridis'
    dynamic_range_db: float = 30  # dB range for visualization

# -----------------------------------------------------------------
# This AzimuthProcessor class is copied from your azimuth.py
# It will use the *real* dsp functions from your mmwave package
# -----------------------------------------------------------------
class AzimuthProcessor:
    def __init__(self, config):
        self.config = config
        self.steering_vec = dsp.gen_steering_vec(
            config.angle_range, 
            config.angle_res,
            config.virt_ant
        )[1] 
    
    def load_and_organize_data(self):
        raw_data = np.load(self.config.input_file, allow_pickle=True)
        print(f"Loaded raw data shape: {raw_data.shape} from {self.config.input_file}")
        
        organized_data = np.apply_along_axis(
            self._organize_iq, 
            1, 
            raw_data,
            num_chirps=self.config.num_chirps,
            num_rx=self.config.num_rx,
            num_samples=self.config.num_samples
        )
        return organized_data
    
    def _organize_iq(self, data, num_chirps, num_rx, num_samples):
        expected_length = num_chirps * num_rx * num_samples * 2
        if data.size != expected_length:
            raise ValueError(f"Data length {data.size} != expected {expected_length}")
        
        data_reshaped = data.reshape(num_chirps, num_rx, num_samples, 2)
        return data_reshaped[..., 0] + 1j * data_reshaped[..., 1]
    
    def process_frame(self, frame):
        # 1. Range FFT
        radar_cube = dsp.range_processing(frame)
        # 2. Static clutter removal
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
        # 5. Generate heatmap
        return self._generate_heatmap(range_azimuth)
    
    def _generate_heatmap(self, range_azimuth):
        heatmap = 10 * np.log10(range_azimuth + 1e-10)
        max_power = np.max(heatmap)
        heatmap = np.clip(heatmap, max_power-self.config.dynamic_range_db, max_power)
        return heatmap

# -----------------------------------------------------------------
# Main function to run the plotting
# -----------------------------------------------------------------
def main():
    print("Generating Radial (Polar) Plot...")
    
    config = RadarConfig()
    
    if not os.path.exists(config.input_file):
        print(f"ERROR: File not found at {config.input_file}")
        print("Please check the 'input_file' variable in the RadarConfig section.")
        return

    processor = AzimuthProcessor(config)
    
    try:
        organized_data = processor.load_and_organize_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    print(f"Processing first frame from {config.input_file}...")
    heatmap_db = processor.process_frame(organized_data[0])
    
    # --- Create the coordinate axes for the plot ---
    angle_axis_deg = np.linspace(-config.angle_range, config.angle_range, heatmap_db.shape[0])
    angle_axis_rad = angle_axis_deg * np.pi / 180.0
    range_axis_m = np.linspace(0, config.range_resolution * config.bins_processed, heatmap_db.shape[1])
    
    # --- Create the plot ---
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 10))
    
    R_grid, Theta_grid = np.meshgrid(range_axis_m, angle_axis_rad)
    c = ax.pcolormesh(Theta_grid, R_grid, heatmap_db, cmap=config.cmap, shading='auto')
    
    # --- Format the plot ---
    ax.set_title(f'Radial Range-Azimuth Plot\n(File: {config.input_file})', pad=20)
    ax.set_theta_zero_location('N') # Set 0-degrees to North
    ax.set_thetamin(-config.angle_range)
    ax.set_thetamax(config.angle_range)
    ax.set_ylabel('Range (m)', position=(0.5, 1.2), labelpad=-30)
    ax.yaxis.set_label_position('left')
    
    cbar = fig.colorbar(c, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Power (dB)')
    
    plt.tight_layout()
    
    output_filename = 'radial_range_azimuth_plot.png'
    plt.savefig(output_filename)
    print(f"Successfully saved plot to '{output_filename}'")
    plt.show() # Also show the plot to your screen

if __name__ == "__main__":
    main()
