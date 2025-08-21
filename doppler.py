import mmwave.dsp as dsp
from mmwave.dsp.utils import Window
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass

@dataclass
class RadarConfig:
    """Configuration parameters for radar processing"""
    # File parameters
    input_file: str = './adc_data_150_2025-08-20_12-57-06.npy'
    
    # Radar parameters
    num_chirps: int = 182 * 3  # 546
    num_rx: int = 4
    num_samples: int = 256
    num_tx: int = 3
    
    # Processing parameters
    range_resolution: float = 0.0488  # meters
    doppler_resolution: float = 0.0356  # m/s
    angle_range: int = 90  # degrees
    angle_res: int = 9.5  # degrees
    virt_ant: int = 12
    
    # Visualization
    cmap: str = 'viridis'
    interval: int = 200  # ms between frames

class RadarProcessor:
    def __init__(self, config):
        self.config = config
        self.steering_vec = dsp.gen_steering_vec(
            config.angle_range, 
            config.angle_res,
            config.virt_ant
        )
        # Initialize window function for range processing
        self.range_window = Window.HANNING
    
    def load_and_organize_data(self):
        """Load and convert raw ADC data to complex format"""
        raw_data = np.load(self.config.input_file, allow_pickle=True)
        print(f"Loaded raw data shape: {raw_data.shape}")
        
        organized_data = np.zeros((raw_data.shape[0], 
                                 self.config.num_chirps, 
                                 self.config.num_rx, 
                                 self.config.num_samples), dtype=np.complex64)
        
        for frame_idx in range(raw_data.shape[0]):
            organized_data[frame_idx] = self._organize_iq(
                raw_data[frame_idx],
                self.config.num_chirps,
                self.config.num_rx,
                self.config.num_samples
            )
        
        print(f"Organized data shape: {organized_data.shape}")
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
        # 1. Range FFT with windowing
        radar_cube = dsp.range_processing(frame, window_type_1d=self.range_window)
        
        # 2. Static clutter removal
        radar_cube = dsp.clutter_removal(radar_cube)
        
        # 3. Doppler processing
        doppler_fft = self._process_doppler(radar_cube)
        
        # 4. Create heatmap (log magnitude)
        return self._generate_heatmap(doppler_fft)
    
    def _process_doppler(self, radar_cube):
        """Handle Doppler processing steps"""
        # Virtual antenna arrangement for TDM MIMO
        radar_cube = np.concatenate(
            [radar_cube[i::self.config.num_tx, ...] 
            for i in range(self.config.num_tx)], 
            axis=1)
        
        # Doppler FFT with shift
        return np.fft.fftshift(
            np.fft.fft(radar_cube, axis=0),
            axes=(0)
        )
    
    def _generate_heatmap(self, doppler_fft):
        """Create visualization-ready heatmap"""
        power = np.abs(doppler_fft.sum(axis=1))
        db_scale = 10 * np.log10(power + 1e-10)
        
        # Dynamic scaling - keep top 30dB
        max_power = np.max(db_scale)
        db_scale = np.clip(db_scale, max_power-30, max_power)
        
        # Mask center
        zero_doppler_idx = db_scale.shape[0] // 2
        db_scale[zero_doppler_idx-1:zero_doppler_idx+1, :] = max_power-30
        
        return db_scale

 

    # def _generate_heatmap(self, doppler_fft):
    #     heatmap = 10 * np.log10(np.abs(doppler_fft.sum(axis=1)))
        
    #     # Mask stationary clutter (zero Doppler)
    #     zero_doppler_idx = heatmap.shape[0] // 2
    #     heatmap[zero_doppler_idx-1:zero_doppler_idx+1, :] = -np.nan  # Hide center
        
    #     return heatmap

    def animate_heatmaps(self, heatmaps):
        """Create animation of processed heatmaps"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate axis limits in physical units
        max_range = self.config.num_samples * self.config.range_resolution
        max_velocity = (self.config.num_chirps // 2) * self.config.doppler_resolution
        
        # Create initial plot
        im = ax.imshow(
            heatmaps[0], 
            aspect='auto', 
            cmap=self.config.cmap,
            interpolation='none',
            extent=[0, max_range, -max_velocity, max_velocity]
        )
        
        # Add colorbar and labels
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power (dB)')
        ax.set_xlabel('Range (m)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title('Range-Doppler Heatmap')

        frame_text = ax.annotate(
            f'Frame: 1/{len(heatmaps)}', 
            xy=(0.02, 0.95), 
            xycoords='axes fraction',
            color='white',
            fontsize=12,
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        # zero_line = ax.axhline(0, color='white', linestyle='--', alpha=0.5)
        # Animation update function
        def update(frame):
            im.set_data(heatmaps[frame])
            frame_text.set_text(f'Frame: {frame+1}/{len(heatmaps)}')
            ax.set_title(f'Range-Doppler Heatmap - Frame {frame+1}')
            return [im,frame_text]
        
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
    processor = RadarProcessor(config)
    
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