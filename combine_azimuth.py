import numpy as np
import matplotlib.pyplot as plt
from mmwave import dsp
from dataclasses import dataclass
import glob
import os

# --- We copy the Config and Processor classes from your azimuth.py ---
# (Slightly modified to not require a file at init)
@dataclass
class RadarConfig:
    """Configuration parameters for radar processing"""
    # File parameters
    input_file: str = '' # We will set this in the loop
    
    # Radar parameters
    num_chirps: int = 182 * 3  # 546
    num_rx: int = 4
    num_samples: int = 256
    num_tx: int = 3
    
    # Processing parameters
    range_resolution: float = 0.0488  # meters
    doppler_resolution: float = 0.0806  # m/s
    angle_range: int = 90  # degrees
    angle_res: int = 1  # degrees
    virt_ant: int = 12
    bins_processed: int = 256
    
    # Visualization
    dynamic_range_db: float = 30  # dB range for visualization

# (Copied from azimuth.py)
class AzimuthProcessor:
    def __init__(self, config):
        self.config = config
        self.steering_vec = dsp.gen_steering_vec(
            config.angle_range, 
            config.angle_res,
            config.virt_ant
        )[1]  # Returns (num_vec, steering_vec)
    
    def load_and_organize_data(self):
        """Load and convert raw ADC data to complex format"""
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
        """Reshape raw ADC data into complex format"""
        expected_length = num_chirps * num_rx * num_samples * 2
        if data.size != expected_length:
            raise ValueError(f"Data length {data.size} != expected {expected_length}")
        
        data_reshaped = data.reshape(num_chirps, num_rx, num_samples, 2)
        return data_reshaped[..., 0] + 1j * data_reshaped[..., 1]
    
    def process_frame(self, frame):
        """Full processing pipeline for one frame"""
        # 1. Range FFT
        radar_cube = dsp.range_processing(frame)
        # 2. Static clutter removal
        radar_cube = dsp.clutter_removal(radar_cube)
        # 3. Virtual antenna arrangement for TDM MIMO
        radar_cube = np.concatenate(
            [radar_cube[i::self.config.num_tx, ...] 
             for i in range(self.config.num_tx)],
            axis=1
        )
        # 4. Initialize arrays for range-azimuth data
        range_azimuth = np.zeros((self.steering_vec.shape[0], self.config.bins_processed))
        
        # 5. Process each range bin with Capon beamforming
        for i in range(self.config.bins_processed):
            range_azimuth[:, i], _ = dsp.aoa_capon(
                radar_cube[:, :, i].T, 
                self.steering_vec, 
                magnitude=True
            )
        
        # 6. Generate heatmap (log magnitude)
        return self._generate_heatmap(range_azimuth)
    
    def _generate_heatmap(self, range_azimuth):
        """Create visualization-ready heatmap"""
        heatmap = 10 * np.log10(range_azimuth + 1e-10)
        max_power = np.max(heatmap)
        heatmap = np.clip(heatmap, max_power-self.config.dynamic_range_db, max_power)
        return heatmap

# --- Main comparison function ---
def main():
    # Define the datasets to process, mapping folder names to their expected angle
    datasets = {
        '0_degree': 0,
        '15_degree_positive': 15,
        '15_degree_negative': -15,
        '30_degree_positive': 30,
        '30_degree_negative': -30
    }
    
    plt.figure(figsize=(14, 8))
    
    for folder, expected_angle in datasets.items():
        # Find the first .npy file in the directory
        try:
            file_path = glob.glob(f'./{folder}/*.npy')[0]
        except IndexError:
            print(f"WARNING: No .npy files found in ./{folder}/. Skipping.")
            continue
            
        print(f"Processing: {file_path} (Expected angle: {expected_angle}°)")
        
        # 1. Initialize config and processor for this file
        config = RadarConfig(input_file=file_path)
        processor = AzimuthProcessor(config)
        
        # 2. Load and organize data
        try:
            organized_data = processor.load_and_organize_data()
        except ValueError as e:
            print(f"Error processing {file_path}: {e}. Skipping.")
            continue
            
        # 3. Process only the first frame to get a heatmap
        heatmap = processor.process_frame(organized_data[0])
        
        # 4. Convert 2D heatmap to 1D angle spectrum
        # We average the power (in dB) across all range bins
        angle_spectrum_db = np.mean(heatmap, axis=1)
        
        # 5. Normalize the 1D spectrum to have a max of 0 dB for easy comparison
        angle_spectrum_normalized = angle_spectrum_db - np.max(angle_spectrum_db)
        
        # 6. Create the angle axis for plotting
        angle_axis = np.linspace(
            -config.angle_range, 
            config.angle_range, 
            len(angle_spectrum_normalized)
        )
        
        # 7. Plot the normalized spectrum
        plt.plot(angle_axis, angle_spectrum_normalized, label=f'Data from {folder}')

    # --- Format the final plot ---
    plt.title('Combined Angle of Arrival Spectrums')
    plt.xlabel('Azimuth Angle (degrees)')
    plt.ylabel('Normalized Power (dB)')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.ylim(-40, 5) # Show top 40 dB of power
    plt.xticks(np.arange(-90, 91, 15)) # Add ticks every 15 degrees
    plt.show()

if __name__ == "__main__":
    main()