import mmWave
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass

@dataclass
class RadarConfig:
    """Configuration parameters for radar processing"""
    # File parameters
    input_file: str = './adc_data_150_2025-08-20_12-52-02.npy'
    
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
    cmap: str = 'viridis'
    interval: int = 200  # ms between frames
    dynamic_range_db: float = 30  # dB range for visualization

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
        # Load raw data
        raw_data = np.load(self.config.input_file, allow_pickle=True)
        print(f"Loaded raw data shape: {raw_data.shape}")
        
        # Organize into complex format
        organized_data = np.apply_along_axis(
            self._organize_iq, 
            1, 
            raw_data,
            num_chirps=self.config.num_chirps,
            num_rx=self.config.num_rx,
            num_samples=self.config.num_samples
        )
        print(f"Organized data shape: {organized_data.shape}")
        return organized_data
    
    def _organize_iq(self, data, num_chirps, num_rx, num_samples):
        """Reshape raw ADC data into complex format"""
        expected_length = num_chirps * num_rx * num_samples * 2
        if data.size != expected_length:
            raise ValueError(f"Data length {data.size} != expected {expected_length}")
        
        # Reshape and convert to complex
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
        beam_weights = np.zeros((self.config.virt_ant, self.config.bins_processed), dtype=np.complex128)
        
        # 5. Process each range bin with Capon beamforming
        for i in range(self.config.bins_processed):
            range_azimuth[:, i], beam_weights[:, i] = dsp.aoa_capon(
                radar_cube[:, :, i].T, 
                self.steering_vec, 
                magnitude=True
            )
        
        # 6. Generate heatmap (log magnitude)
        return self._generate_heatmap(range_azimuth)
    
    def _generate_heatmap(self, range_azimuth):
        """Create visualization-ready heatmap"""
        # Convert to dB scale
        heatmap = 10 * np.log10(range_azimuth + 1e-10)
        
        # Apply dynamic scaling
        max_power = np.max(heatmap)
        heatmap = np.clip(heatmap, max_power-self.config.dynamic_range_db, max_power)
        
        return heatmap

    def animate_heatmaps(self, heatmaps):
        """Create animation of processed heatmaps"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate axis limits in physical units
        max_range = self.config.bins_processed * self.config.range_resolution
        angle_axis = np.linspace(
            -self.config.angle_range, 
            self.config.angle_range, 
            self.steering_vec.shape[0]
        )
        
        # Create initial plot
        im = ax.imshow(
            heatmaps[0], 
            aspect='auto', 
            cmap=self.config.cmap,
            extent=[0, max_range, -self.config.angle_range, self.config.angle_range],
            origin='lower'
        )
        
        # Add colorbar and labels
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power (dB)')
        ax.set_xlabel('Range (m)')
        ax.set_ylabel('Azimuth Angle (deg)')
        ax.set_title('Range-Azimuth Heatmap')

        # Add frame counter annotation
        frame_text = ax.annotate(
            f'Frame: 1/{len(heatmaps)}', 
            xy=(0.02, 0.95), 
            xycoords='axes fraction',
            color='white',
            fontsize=12,
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        
        # Animation update function
        def update(frame):
            im.set_data(heatmaps[frame])
            frame_text.set_text(f'Frame: {frame+1}/{len(heatmaps)}')
            
            # Update title with current frame info
            ax.set_title(f'Range-Azimuth Heatmap - Frame {frame+1}/{len(heatmaps)}')
            
            return [im, frame_text]
        
        # Create animation
        ani = FuncAnimation(
            fig, 
            update, 
            frames=len(heatmaps),
            interval=self.config.interval, 
            blit=True
        )
        
        plt.tight_layout()
        plt.show()
        return ani

def main():
    # Initialize configuration and processor
    config = RadarConfig()
    processor = AzimuthProcessor(config)
    
    # Load and organize data
    organized_data = processor.load_and_organize_data()
    
    # Process all frames with progress indication
    print("Processing frames...")
    heatmaps = []
    for i, frame in enumerate(organized_data):
        heatmaps.append(processor.process_frame(frame))
        print(f"Processed frame {i+1}/{len(organized_data)}", end='\r')
    
    # Animate results
    print("\nGenerating animation...")
    processor.animate_heatmaps(heatmaps)

if __name__ == "__main__":
    main()