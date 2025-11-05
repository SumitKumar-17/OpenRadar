from mmwave import dsp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter  # Smooths noise
import os

# -------------------- Configuration --------------------
@dataclass
class RadarConfig:
    """Configuration parameters for radar processing"""
    # File parameters
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

    # Visualization and filters
    noise_filter_size: int = 3
    cmap: str = 'viridis'
    interval: int = 200
    dynamic_range_db: float = 40


# -------------------- Azimuth Processor --------------------
class AzimuthProcessor:
    def __init__(self, config):
        self.config = config
        # Precompute steering vector for AoA estimation
        self.steering_vec = dsp.gen_steering_vec(
            config.angle_range,
            config.angle_res,
            config.virt_ant
        )[1]

    # ✅ Updated IQ organization using the verified logic
    def _organize_iq(self, raw_frame, num_chirps, num_rx, num_samples):
        """
        Reorganizes raw ADC data into (num_chirps, num_rx, num_samples)
        following IWR1843 / DCA1000 IQ interleaving pattern
        """
        ret = np.zeros(len(raw_frame) // 2, dtype=complex)

        # Separate IQ components (interleaved I-Q from 4 channels)
        ret[0::2] = raw_frame[0::4] + 1j * raw_frame[2::4]
        ret[1::2] = raw_frame[1::4] + 1j * raw_frame[3::4]

        # Reshape into radar cube: [chirp, rx, samples]
        return ret.reshape((num_chirps, num_rx, num_samples))

    # -------------------- Frame Processing --------------------
    def process_frame(self, frame):
        """Full processing pipeline for one frame"""
        # Step 1: Range FFT
        radar_cube = dsp.range_processing(frame)
        # Step 2: Static clutter removal
        radar_cube = dsp.clutter_removal(radar_cube)
        # Step 3: Virtual antenna combination
        radar_cube = np.concatenate(
            [radar_cube[i::self.config.num_tx, ...]
             for i in range(self.config.num_tx)],
            axis=1
        )
        # Step 4: Capon Beamforming (Angle FFT)
        range_azimuth = np.zeros((self.steering_vec.shape[0], self.config.bins_processed))
        for i in range(self.config.bins_processed):
            range_azimuth[:, i], _ = dsp.aoa_capon(
                radar_cube[:, :, i].T,
                self.steering_vec,
                magnitude=True
            )

        # Step 5: Apply Gaussian smoothing (noise reduction)
        if self.config.noise_filter_size > 1:
            range_azimuth = gaussian_filter(range_azimuth, sigma=1)
jpg
        # Step 6: Log scale conversion (dB)
        return self._generate_heatmap(range_azimuth)

    def _generate_heatmap(self, range_azimuth):
        """Generate log-magnitude heatmap (for visualization)"""
        heatmap = 10 * np.log10(range_azimuth + 1e-10)
        max_power = np.max(heatmap)
        heatmap = np.clip(heatmap, max_power - self.config.dynamic_range_db, max_power)
        return heatmap

    # -------------------- Data Loading --------------------
    def load_and_organize_data(self):
        """Load raw ADC .npy data and convert to complex frame sequence"""
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

    # -------------------- Heatmap Animation --------------------
    def animate_heatmaps(self, heatmaps, save_gif=True):
        """
        Animation with target tracking, direction detection, and lateral speed estimate.
        Fixed version — avoids modifying read-only Axes.texts.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        max_range = self.config.bins_processed * self.config.range_resolution
        extent = [0, max_range, -self.config.angle_range, self.config.angle_range]

        im = ax.imshow(
            heatmaps[0], aspect='auto', cmap=self.config.cmap,
            extent=extent, origin='lower'
        )
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power (dB)')
        ax.set_xlabel('Range (m)')
        ax.set_ylabel('Azimuth Angle (deg)')

        frame_text = ax.annotate(
            f'Frame: 1/{len(heatmaps)}', xy=(0.02, 0.95), xycoords='axes fraction',
            color='white', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5)
        )
        info_text = ax.annotate(
            "", xy=(0.98, 0.95), xycoords='axes fraction',
            ha='right', va='top', color='white', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5)
        )

        (dot,) = ax.plot([], [], 'ro', markersize=6)
        (centroid_dot,) = ax.plot([], [], 'yo', markersize=6)
        traj_line, = ax.plot([], [], '-', linewidth=1)

        traj_ranges, traj_az, traj_x, traj_t = [], [], [], []
        dt = self.config.interval / 1000.0

        azimuths = np.linspace(-self.config.angle_range, self.config.angle_range, heatmaps[0].shape[0])
        ranges = np.linspace(0, max_range, heatmaps[0].shape[1])

        def update(frame_idx):
            heatmap = heatmaps[frame_idx]
            im.set_data(heatmap)
            frame_text.set_text(f'Frame: {frame_idx+1}/{len(heatmaps)}')
            ax.set_title(f'Range–Azimuth Heatmap - Frame {frame_idx+1}')

            # ---- centroid azimuth ----
            p_az = np.sum(heatmap, axis=1)
            centroid_az = np.sum(azimuths * p_az) / np.sum(p_az) if np.sum(p_az) > 0 else 0.0

            # ---- strongest reflection ----
            az_idx, range_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            peak_az = azimuths[az_idx]
            peak_r = ranges[range_idx]

            # ---- store trajectory ----
            traj_ranges.append(peak_r)
            traj_az.append(centroid_az)
            traj_t.append(len(traj_t) * dt)
            x = peak_r * np.sin(np.deg2rad(centroid_az))
            traj_x.append(x)

            # ---- compute direction ----
            # direction = "N/A"
            # speed_lateral = 0.0
            # if len(traj_x) >= 2:
            #     dx = traj_x[-1] - traj_x[-2]
            #     speed_lateral = dx / dt
            #     if abs(speed_lateral) < 0.05:
            #         direction = "Stationary"
            #     elif speed_lateral > 0:
            #         direction = "Left → Right"
            #     else:
            #         direction = "Right → Left"

            # ---- update plot markers ----
            # dot.set_data([peak_r], [peak_az])
            # centroid_dot.set_data([peak_r], [centroid_az])
            # traj_line.set_data(traj_ranges, traj_az)

            # ---- update text instead of reassigning ax.texts ----
            info_text.set_text(
                # f"Dir: {direction} | Lat speed: {speed_lateral:.2f} m/s\n"
                f"Peak: {peak_r:.2f} m, {peak_az:.1f}°"
            )

            print(f"Frame {frame_idx+1:03d}: centroid_az={centroid_az:.2f}°, "
                f"peak at r={peak_r:.2f}m, az={peak_az:.1f}°, "
                # f"lat_speed={speed_lateral:.2f} m/s, dir={direction}"
                )

            # return [im, frame_text, info_text, dot, centroid_dot, traj_line]
            return [im, frame_text]

        ani = FuncAnimation(
            fig, update, frames=len(heatmaps),
            interval=self.config.interval, blit=True
        )
        if save_gif:
            output_dir = os.path.dirname(self.config.input_file)
            gif_path = os.path.join(output_dir, "0_deg_pos2.gif")
            print(f"\nSaving GIF to: {gif_path}")
            writer = PillowWriter(fps=int(1000 / self.config.interval))
            ani.save(gif_path, writer=writer)
            print("GIF saved successfully ✅")

        plt.tight_layout()
        plt.show()
        return ani



    # def animate_heatmaps(self, heatmaps):
    #     """Create animation of processed Range–Azimuth heatmaps"""
    #     fig, ax = plt.subplots(figsize=(12, 6))
    #     max_range = self.config.bins_processed * self.config.range_resolution
    #     extent = [0, max_range, -self.config.angle_range, self.config.angle_range]

    #     im = ax.imshow(
    #         heatmaps[0], aspect='auto', cmap=self.config.cmap,
    #         extent=extent, origin='lower'
    #     )
    #     cbar = plt.colorbar(im, ax=ax)
    #     cbar.set_label('Power (dB)')
    #     ax.set_xlabel('Range (m)')
    #     ax.set_ylabel('Azimuth Angle (deg)')

    #     # Frame annotation and target marker
    #     frame_text = ax.annotate(
    #         f'Frame: 1/{len(heatmaps)}', xy=(0.02, 0.95), xycoords='axes fraction',
    #         color='white', fontsize=12,
    #         bbox=dict(boxstyle='round', facecolor='black', alpha=0.5)
    #     )
    #     (dot,) = ax.plot([], [], 'ro', markersize=6)

    #     def update(frame):
    #         heatmap = heatmaps[frame]
    #         im.set_data(heatmap)
    #         frame_text.set_text(f'Frame: {frame + 1}/{len(heatmaps)}')
    #         ax.set_title(f'Range–Azimuth Heatmap - Frame {frame + 1}')

    #         # Target detection
    #         az_idx, range_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    #         azimuths = np.linspace(-self.config.angle_range, self.config.angle_range, heatmap.shape[0])
    #         ranges = np.linspace(0, max_range, heatmap.shape[1])
    #         az_deg = azimuths[az_idx]
    #         r_m = ranges[range_idx]

    #         print(f"Frame {frame + 1:03d}: Strongest reflection at {r_m:.2f} m, {az_deg:.1f}°")

    #         # ✅ FIX: pass as list to avoid “x must be a sequence” error
    #         dot.set_data([r_m], [az_deg])

    #         return [im, frame_text, dot]

    #     ani = FuncAnimation(fig, update, frames=len(heatmaps), interval=self.config.interval, blit=True)
    #     plt.tight_layout()
    #     plt.show()
    #     return ani


# -------------------- Main --------------------
def main():
    config = RadarConfig()
    if 'your_file_goes_here' in config.input_file:
        print("STOP: Please update the 'input_file' variable in RadarConfig.")
        return

    processor = AzimuthProcessor(config)
    organized_data = processor.load_and_organize_data()

    print("Processing frames...")
    heatmaps = [processor.process_frame(frame) for frame in organized_data]

    print("\nGenerating animation with target tracking...")
    processor.animate_heatmaps(heatmaps, save_gif=True)


if __name__ == "__main__":
    main()
