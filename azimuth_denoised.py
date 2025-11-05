from mmwave import dsp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from scipy.signal import medfilt2d  
@dataclass
class RadarConfig:
    """Configuration parameters for radar processing"""
    # File parameters
    input_file: str = './30_degree_negative/30_degree_negative1_2025-10-16_17-09-04.npy'
    
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
    
    # --- NEW: Noise Removal Parameter ---
    # The kernel size for the median filter. Must be an odd number (e.g., 3, 5).
    # A larger number means more smoothing/noise removal. 3 is a good start.
    noise_filter_size: int = 5
    
    # Visualization
    cmap: str = 'viridis'
    interval: int = 200
    dynamic_range_db: float = 25

class AzimuthProcessor:
    def __init__(self, config):
        self.config = config
        self.steering_vec = dsp.gen_steering_vec(
            config.angle_range, 
            config.angle_res,
            config.virt_ant
        )[1]
    
    def _organize_iq(self, data, num_chirps, num_rx, num_samples):
        """Reshape raw ADC data into complex format"""
        expected_length = num_chirps * num_rx * num_samples * 2
        if data.size != expected_length:
            raise ValueError(f"Data length {data.size} != expected {expected_length}")
        data_reshaped = data.reshape(num_chirps, num_rx, num_samples, 2)
        return data_reshaped[..., 0] + 1j * data_reshaped[..., 1]
    
    def process_frame(self, frame):
        """Full processing pipeline for one frame"""
        # Steps 1-3: Standard processing
        radar_cube = dsp.range_processing(frame)
        radar_cube = dsp.clutter_removal(radar_cube)
        radar_cube = np.concatenate(
            [radar_cube[i::self.config.num_tx, ...] 
             for i in range(self.config.num_tx)],
            axis=1
        )
        
        # Step 4: Capon beamforming
        range_azimuth = np.zeros((self.steering_vec.shape[0], self.config.bins_processed))
        for i in range(self.config.bins_processed):
            range_azimuth[:, i], _ = dsp.aoa_capon(
                radar_cube[:, :, i].T, 
                self.steering_vec, 
                magnitude=True
            )
        
        # --- NEW: Step 5: Apply the 2D Median Filter for Noise Removal ---
        # This is applied to the linear power data before converting to dB
        if self.config.noise_filter_size > 1:
            range_azimuth = medfilt2d(range_azimuth, kernel_size=self.config.noise_filter_size)
        # -----------------------------------------------------------------

        # Step 6: Generate heatmap (log magnitude)
        return self._generate_heatmap(range_azimuth)
    
    def _generate_heatmap(self, range_azimuth):
        """Create visualization-ready heatmap"""
        heatmap = 10 * np.log10(range_azimuth + 1e-10)
        max_power = np.max(heatmap)
        heatmap = np.clip(heatmap, max_power-self.config.dynamic_range_db, max_power)
        return heatmap

    # The rest of the functions (load_and_organize_data, animate_heatmaps)
    # and the main() function remain exactly the same as your original script.
    
    def load_and_organize_data(self):
        """Load and convert raw ADC data to complex format"""
        raw_data = np.load(self.config.input_file, allow_pickle=True)
        print(f"Loaded raw data shape: {raw_data.shape}")
        organized_data = np.apply_along_axis(
            self._organize_iq, 1, raw_data,
            num_chirps=self.config.num_chirps,
            num_rx=self.config.num_rx,
            num_samples=self.config.num_samples
        )
        print(f"Organized data shape: {organized_data.shape}")
        return organized_data

    def animate_heatmaps(self, heatmaps):
        """Create animation of processed heatmaps"""
        fig, ax = plt.subplots(figsize=(12, 6))
        max_range = self.config.bins_processed * self.config.range_resolution
        im = ax.imshow(
            heatmaps[0], aspect='auto', cmap=self.config.cmap,
            extent=[0, max_range, -self.config.angle_range, self.config.angle_range],
            origin='lower'
        )
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power (dB)')
        ax.set_xlabel('Range (m)')
        ax.set_ylabel('Azimuth Angle (deg)')
        frame_text = ax.annotate(
            f'Frame: 1/{len(heatmaps)}', xy=(0.02, 0.95), xycoords='axes fraction',
            color='white', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        def update(frame):
            im.set_data(heatmaps[frame])
            frame_text.set_text(f'Frame: {frame+1}/{len(heatmaps)}')
            ax.set_title(f'Range-Azimuth Heatmap - Frame {frame+1}/{len(heatmaps)}')
            return [im, frame_text]
        
        ani = FuncAnimation(
            fig, update, frames=len(heatmaps),
            interval=self.config.interval, blit=True
        )
        plt.tight_layout()
        plt.show()
        return ani

def main():
    config = RadarConfig()
    if 'your_file_goes_here' in config.input_file:
         print("STOP: Please update the 'input_file' variable in the RadarConfig section of the script.")
         return
    processor = AzimuthProcessor(config)
    organized_data = processor.load_and_organize_data()
    print("Processing frames...")
    heatmaps = [processor.process_frame(frame) for frame in organized_data]
    print("\nGenerating animation...")
    processor.animate_heatmaps(heatmaps)

if __name__ == "__main__":
    main()