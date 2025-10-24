import numpy as np
import matplotlib.pyplot as plt
from mmwave import dsp
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass

# --- Configuration ---
@dataclass
class RadarConfig:
    """Configuration parameters for radar processing"""
    # --- STEP 1: MODIFY THIS SECTION ---
    # Update this to the path of the .npy file you want to animate
    input_file: str = './0_degree/0_degree_positive1_2025-10-16_15-25-11.npy'

    # Radar parameters from your original script
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

    # Animation parameters
    cmap: str = 'viridis'
    interval: int = 100  # ms between frames (faster animation)
    dynamic_range_db: float = 30

class AzimuthAnimator:
    def __init__(self, config):
        self.config = config
        self.steering_vec = dsp.gen_steering_vec(
            config.angle_range,
            config.angle_res,
            config.virt_ant
        )[1]

    def _organize_iq(self, data):
        """Reshape raw ADC data into complex format for a single frame."""
        num_chirps, num_rx, num_samples = self.config.num_chirps, self.config.num_rx, self.config.num_samples
        expected_length = num_chirps * num_rx * num_samples * 2
        if data.size != expected_length:
            raise ValueError(f"Data length {data.size} != expected {expected_length}")
        data_reshaped = data.reshape(num_chirps, num_rx, num_samples, 2)
        return data_reshaped[..., 0] + 1j * data_reshaped[..., 1]

    def process_frame(self, frame_data):
        """Full processing pipeline for one frame to generate a heatmap."""
        # 1. Range FFT
        radar_cube = dsp.range_processing(frame_data)
        # 2. Clutter Removal
        radar_cube = dsp.clutter_removal(radar_cube)
        # 3. Virtual Antenna Arrangement
        radar_cube_va = np.concatenate(
            [radar_cube[i::self.config.num_tx, ...] for i in range(self.config.num_tx)],
            axis=1
        )
        # 4. Capon Beamforming across all range bins
        range_azimuth = np.zeros((self.steering_vec.shape[0], self.config.bins_processed))
        for i in range(self.config.bins_processed):
            range_azimuth[:, i], _ = dsp.aoa_capon(
                radar_cube_va[:, :, i].T, self.steering_vec, magnitude=True
            )
        # 5. Convert to dB for visualization
        heatmap = 10 * np.log10(range_azimuth + 1e-10)
        max_power = np.max(heatmap)
        return np.clip(heatmap, max_power - self.config.dynamic_range_db, max_power)

    def run_animation(self):
        """Loads data, processes all frames, and creates the animation."""
        try:
            raw_data = np.load(self.config.input_file, allow_pickle=True)
            print(f"Loaded {raw_data.shape[0]} frames from '{self.config.input_file}'")
        except FileNotFoundError:
            print(f"ERROR: File not found at '{self.config.input_file}'. Please check the path.")
            return

        # Process all frames first
        print("Processing all frames...")
        heatmaps = [self.process_frame(self._organize_iq(frame)) for frame in raw_data]
        print("Processing complete. Starting animation...")

        # --- Setup the plot ---
        fig, ax = plt.subplots(figsize=(12, 7))
        max_range = self.config.bins_processed * self.config.range_resolution
        im = ax.imshow(
            heatmaps[0],
            aspect='auto',
            cmap=self.config.cmap,
            extent=[0, max_range, -self.config.angle_range, self.config.angle_range],
            origin='lower'
        )
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power (dB)')
        ax.set_xlabel('Range (m)')
        ax.set_ylabel('Azimuth Angle (Degrees)')

        # Animation update function
        def update(frame):
            im.set_data(heatmaps[frame])
            ax.set_title(f'Range-Azimuth Heatmap - Frame {frame + 1}/{len(heatmaps)}')
            return [im]

        # Create and show the animation
        ani = FuncAnimation(
            fig,
            update,
            frames=len(heatmaps),
            interval=self.config.interval,
            blit=True
        )
        plt.show()

def main():
    """Main function to run the animator."""
    config = RadarConfig()
    if 'your_file_goes_here' in config.input_file:
         print("STOP: Please update the 'input_file' variable in the RadarConfig section of the script.")
         return
    animator = AzimuthAnimator(config)
    animator.run_animation()

if __name__ == "__main__":
    main()