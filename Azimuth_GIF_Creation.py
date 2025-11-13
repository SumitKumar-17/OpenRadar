import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from dataclasses import dataclass
import cv2
from PIL import Image
import argparse
from scipy import signal

@dataclass
class RadarConfig:
    """Configuration parameters for radar processing"""
    # File parameters (hardcoded - replace with your paths)
    input_file: str = '/home/dadi/Downloads/MultiplePeople_6_26ndAug_500_2025-08-26_16-16-29.npy'  # Your radar .npy file
    video_file: str = '/home/dadi/Downloads/document_6197368204440902417.mp4'  # Your video .mp4 file
    
    # Radar parameters
    num_chirps: int = 182 * 3
    num_rx: int = 4
    num_samples: int = 256
    num_tx: int = 3
    radar_fps: float = 10.0  # Radar frame rate
    
    # Video parameters
    video_fps: float = 29.0  # Video frame rate
    
    # Processing parameters
    range_resolution: float = 0.0488
    doppler_resolution: float = 0.0806  # Not used in azimuth, but kept for consistency
    angle_range: int = 90
    angle_res: int = 1
    virt_ant: int = 12
    bins_processed: int = 256
    
    # Visualization
    cmap: str = 'viridis'
    
    # Frame ranges (radar set via CLI, video calculated)
    radar_start_frame: int = 0
    radar_end_frame: int = 0
    video_start_frame: int = 0
    video_end_frame: int = 0
    
    # Output
    output_gif: str = 'azimuth_clip.gif'
    gif_fps: int = 10  # Output GIF frame rate

class AzimuthProcessor:
    def __init__(self, config):
        self.config = config
        # Generate steering vector (assuming ULA with d=0.5 lambda)
        angles_deg = np.arange(-self.config.angle_range, self.config.angle_range + self.config.angle_res, self.config.angle_res)
        self.angles_deg = angles_deg
        angles_rad = angles_deg * np.pi / 180
        self.steering_vec = np.exp(-1j * np.pi * np.arange(self.config.virt_ant)[:, np.newaxis] * np.sin(angles_rad))
        # Range window
        self.range_window = signal.windows.hann(self.config.num_samples)
    
    def extract_video_frames(self):
        """Extract specific video frames"""
        cap = cv2.VideoCapture(self.config.video_file)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.config.video_file}")
        
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video total frames: {total_video_frames}")
        print(f"Extracting video frames: {self.config.video_start_frame} to {self.config.video_end_frame}")
        
        if self.config.video_end_frame >= total_video_frames:
            raise ValueError(f"Video end frame {self.config.video_end_frame} exceeds total frames {total_video_frames}")
        
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.config.video_start_frame)
        
        for frame_idx in range(self.config.video_start_frame, self.config.video_end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
            if len(frames) % 50 == 0:
                print(f"Extracted {len(frames)} video frames", end='\r')
        
        cap.release()
        print(f"\nExtracted {len(frames)} video frames")
        return frames
    
    def load_and_process_radar_frames(self):
        """Load and process specific radar frames"""
        print(f"Processing radar frames: {self.config.radar_start_frame} to {self.config.radar_end_frame}")
        
        raw_data = np.load(self.config.input_file, allow_pickle=True)
        print(f"Radar data total frames: {len(raw_data)}")
        
        radar_total_frames = len(raw_data)
        if self.config.radar_end_frame >= radar_total_frames:
            raise ValueError(f"Radar end frame {self.config.radar_end_frame} exceeds total frames {radar_total_frames}")
        
        heatmaps = []
        for frame_idx in range(self.config.radar_start_frame, self.config.radar_end_frame + 1):
            if frame_idx < len(raw_data):
                organized_frame = self._organize_frame(
                    raw_data[frame_idx],
                    self.config.num_chirps,
                    self.config.num_rx,
                    self.config.num_samples
                )
                heatmap = self.process_frame(organized_frame)
                heatmaps.append(heatmap)
                if len(heatmaps) % 10 == 0:
                    print(f"Processed {len(heatmaps)} radar frames", end='\r')
            else:
                print(f"Warning: Radar frame {frame_idx} exceeds available data")
                break
        
        print(f"\nProcessed {len(heatmaps)} radar frames")
        return heatmaps
    
    def _organize_frame(self, data, num_chirps, num_rx, num_samples):
        """Organize single radar frame"""
        expected_length = num_chirps * num_rx * num_samples * 2
        if data.size != expected_length:
            raise ValueError(f"Data length {data.size} != expected {expected_length}")
        
        data_reshaped = data.reshape(num_chirps, num_rx, num_samples, 2)
        return data_reshaped[..., 0] + 1j * data_reshaped[..., 1]
    
    def process_frame(self, frame):
        """Process single radar frame for range-azimuth heatmap"""
        # Range processing
        radar_cube = self._range_processing(frame)
        # Clutter removal
        radar_cube = self._clutter_removal(radar_cube)
        # Virtual antenna arrangement for TDM MIMO
        radar_cube = np.concatenate(
            [radar_cube[i::self.config.num_tx, ...] for i in range(self.config.num_tx)], 
            axis=1
        )  # Shape: (num_chirps_per_tx, virt_ant, num_samples)
        # Process each range bin
        num_angles = self.steering_vec.shape[1]
        range_azimuth = np.zeros((num_angles, self.config.bins_processed))
        for i in range(self.config.bins_processed):
            input_mat = radar_cube[:, :, i].T  # (virt_ant, num_snapshots)
            spectrum, _ = self._aoa_capon(input_mat, self.steering_vec)
            range_azimuth[:, i] = spectrum
        # Generate heatmap
        return self._generate_heatmap(range_azimuth)
    
    def _range_processing(self, frame):
        """Apply windowing and FFT along range (sample) dimension"""
        windowed = frame * self.range_window[np.newaxis, np.newaxis, :]
        radar_cube = np.fft.fft(windowed, axis=2)
        return radar_cube
    
    def _clutter_removal(self, radar_cube):
        """Remove static clutter by subtracting mean across chirps"""
        mean_clutter = np.mean(radar_cube, axis=0, keepdims=True)
        return radar_cube - mean_clutter
    
    def _aoa_capon(self, input_mat, steering_vec):
        """Capon beamformer for AoA estimation"""
        num_snapshots = input_mat.shape[1]
        virt_ant = input_mat.shape[0]
        Rxx = (input_mat @ np.conj(input_mat).T) / num_snapshots
        Rxx_inv = np.linalg.pinv(Rxx + 1e-10 * np.eye(virt_ant))
        num_angles = steering_vec.shape[1]
        spectrum = np.zeros(num_angles)
        for k in range(num_angles):
            v = self.steering_vec[:, k][:, np.newaxis]
            denom = np.real(np.conj(v).T @ Rxx_inv @ v)
            spectrum[k] = 1 / denom if denom > 1e-10 else 0
        # Beam weights not needed, return None
        return spectrum, None
    
    def _generate_heatmap(self, range_azimuth):
        """Create visualization-ready heatmap"""
        heatmap = 10 * np.log10(range_azimuth + 1e-10)
        max_power = np.max(heatmap)
        heatmap = np.clip(heatmap, max_power - 30, max_power)
        return heatmap

    def create_combined_gif(self, heatmaps, video_frames):
        """Create combined GIF with synchronized frames"""
        radar_frame_count = len(heatmaps)
        video_frame_count = len(video_frames)
        
        print(f"Radar frames: {radar_frame_count}, Video frames: {video_frame_count}")
        
        video_sample_indices = np.linspace(0, video_frame_count - 1, radar_frame_count, dtype=int)
        sampled_video_frames = [video_frames[i] for i in video_sample_indices]
        
        print(f"Sampled {len(sampled_video_frames)} video frames to match radar")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Setup radar plot
        max_range = self.config.bins_processed * self.config.range_resolution
        im_radar = ax1.imshow(
            heatmaps[0], 
            aspect='auto', 
            cmap=self.config.cmap,
            extent=[0, max_range, -self.config.angle_range, self.config.angle_range],
            origin='lower'
        )
        ax1.set_xlabel('Range (m)')
        ax1.set_ylabel('Azimuth Angle (deg)')
        ax1.set_title('Range-Azimuth Heatmap (10 FPS)')
        plt.colorbar(im_radar, ax=ax1, label='Power (dB)')
        
        # Setup video plot
        im_video = ax2.imshow(sampled_video_frames[0])
        ax2.axis('off')
        ax2.set_title('Video Ground Truth (29 FPS sampled to 10 FPS)')
        
        # Add frame info
        radar_frame_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, color='white', 
                                    fontsize=12, bbox=dict(facecolor='black', alpha=0.7))
        
        video_frame_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, color='white',
                                    fontsize=12, bbox=dict(facecolor='black', alpha=0.7))
        
        time_text = ax1.text(0.02, 0.88, '', transform=ax1.transAxes, color='white',
                             fontsize=10, bbox=dict(facecolor='black', alpha=0.7))
        
        def update(frame_idx):
            im_radar.set_data(heatmaps[frame_idx])
            im_video.set_data(sampled_video_frames[frame_idx])
            
            radar_frame_num = self.config.radar_start_frame + frame_idx
            video_frame_num = self.config.video_start_frame + video_sample_indices[frame_idx]
            current_time = frame_idx / self.config.radar_fps
            
            radar_frame_text.set_text(f'Radar Frame: {radar_frame_num}')
            video_frame_text.set_text(f'Video Frame: {video_frame_num}')
            time_text.set_text(f'Time: {current_time:.1f}s')
            
            return [im_radar, im_video, radar_frame_text, video_frame_text, time_text]
        
        ani = FuncAnimation(
            fig, 
            update, 
            frames=radar_frame_count,
            interval=1000/self.config.gif_fps,
            blit=True
        )
        
        print("Saving combined GIF...")
        ani.save(
            self.config.output_gif,
            writer=PillowWriter(fps=self.config.gif_fps),
            dpi=100,
            savefig_kwargs={'facecolor': 'white'}
        )
        
        plt.close()
        print(f"Saved: {self.config.output_gif}")

def main():
    parser = argparse.ArgumentParser(description="Generate synchronized azimuth radar and video GIF")
    parser.add_argument('--radar_start', type=int, required=True, help='Radar start frame number')
    parser.add_argument('--radar_end', type=int, required=True, help='Radar end frame number')
    parser.add_argument('--output_gif', type=str, default='azimuth_multiple6.gif', help='Output GIF file name')
    
    args = parser.parse_args()
    
    raw_data = np.load(RadarConfig.input_file, allow_pickle=True)
    radar_total_frames = len(raw_data)
    
    if args.radar_start < 0 or args.radar_start >= radar_total_frames:
        raise ValueError(f"Radar start frame {args.radar_start} invalid (0 to {radar_total_frames-1})")
    if args.radar_end < args.radar_start or args.radar_end >= radar_total_frames:
        raise ValueError(f"Radar end frame {args.radar_end} invalid ({args.radar_start} to {radar_total_frames-1})")
    
    fps_ratio = 29.0 / 10.0
    video_start = round(args.radar_start * fps_ratio)
    video_end = round(args.radar_end * fps_ratio)
    
    config = RadarConfig(
        radar_start_frame=args.radar_start,
        radar_end_frame=args.radar_end,
        video_start_frame=video_start,
        video_end_frame=video_end,
        output_gif=args.output_gif
    )
    
    cap = cv2.VideoCapture(config.video_file)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {config.video_file}")
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if video_start < 0 or video_start >= total_video_frames:
        raise ValueError(f"Video start {video_start} invalid (0 to {total_video_frames-1})")
    if video_end < video_start or video_end >= total_video_frames:
        raise ValueError(f"Video end {video_end} invalid ({video_start} to {total_video_frames-1})")
    
    print(f"Calculated video frames: {video_start} to {video_end}")
    
    processor = AzimuthProcessor(config)
    
    print("Extracting video frames...")
    video_frames = processor.extract_video_frames()
    
    print("Processing radar frames...")
    heatmaps = processor.load_and_process_radar_frames()
    
    print("Creating combined GIF...")
    processor.create_combined_gif(heatmaps, video_frames)
    
    print("\n" + "="*50)
    print("GIF CREATION COMPLETE!")
    print("="*50)
    print(f"Output file: {config.output_gif}")
    print(f"Radar frames: {config.radar_start_frame}-{config.radar_end_frame}")
    print(f"Video frames: {config.video_start_frame}-{config.video_end_frame}")
    print(f"Duration: {(config.radar_end_frame - config.radar_start_frame + 1) / config.radar_fps:.1f} seconds")

if __name__ == "__main__":
    main()