import numpy as np
import matplotlib.pyplot as plt
from mmwave import dsp
from mmwave.dsp.utils import Window
from dataclasses import dataclass
from scipy.optimize import curve_fit

# --- Configuration ---
@dataclass
class RadarConfig:
    """Configuration parameters for radar processing"""
    # Radar parameters from your scripts
    num_chirps: int = 182 * 3
    num_rx: int = 4
    num_samples: int = 256
    num_tx: int = 3
    
    # Processing parameters
    range_resolution: float = 0.0488
    doppler_resolution: float = 0.0356 # Use the one from your doppler script
    
    # Target parameters
    expected_range_m: float = 4.5  # Expected distance of the person
    range_search_window_m: float = 1.0 # Search +/- 1.0m around the expected range

def organize_iq(data, num_chirps, num_rx, num_samples):
    """Reshape raw ADC data into complex format for a single frame."""
    expected_length = num_chirps * num_rx * num_samples * 2
    if data.size != expected_length:
        raise ValueError(f"Data length {data.size} != expected {expected_length}")
    
    data_reshaped = data.reshape(num_chirps, num_rx, num_samples, 2)
    return data_reshaped[..., 0] + 1j * data_reshaped[..., 1]

def process_and_extract_velocity(filepath, config):
    """
    Loads a single .npy file, processes it to find the Range-Doppler map,
    and extracts the peak velocity of the target.
    """
    # 1. Load and organize data
    raw_data = np.load(filepath, allow_pickle=True)
    # We'll process the middle frame for a stable measurement
    middle_frame_idx = raw_data.shape[0] // 2
    frame_data = organize_iq(
        raw_data[middle_frame_idx], 
        config.num_chirps, 
        config.num_rx, 
        config.num_samples
    )
    
    # 2. Range FFT
    radar_cube = dsp.range_processing(frame_data, window_type_1d=Window.HANNING)
    
    # 3. Static Clutter Removal
    radar_cube = dsp.clutter_removal(radar_cube)
    
    # 4. Doppler FFT
    # Arrange virtual antennas
    radar_cube_va = np.concatenate(
        [radar_cube[i::config.num_tx, ...] for i in range(config.num_tx)], 
        axis=1
    )
    doppler_fft = np.fft.fftshift(np.fft.fft(radar_cube_va, axis=0), axes=(0))

    # 5. Create Power Heatmap
    power_heatmap = np.abs(doppler_fft.sum(axis=1))
    
    # 6. Find Peak Velocity in the Expected Range Window
    # Convert meters to range bins
    center_bin = int(config.expected_range_m / config.range_resolution)
    window_bins = int(config.range_search_window_m / config.range_resolution)
    start_bin = max(0, center_bin - window_bins)
    end_bin = min(config.num_samples, center_bin + window_bins)
    
    # Isolate the heatmap to the range of interest
    range_slice = power_heatmap[:, start_bin:end_bin]
    
    # Find the index of the maximum power in this slice
    if range_slice.size == 0:
        print(f"Warning: No data in the specified range for {filepath}")
        return None
        
    peak_doppler_idx, _ = np.unravel_index(np.argmax(range_slice), range_slice.shape)
    
    # 7. Convert Doppler index to physical velocity
    num_doppler_bins = doppler_fft.shape[0]
    velocity = (peak_doppler_idx - num_doppler_bins / 2) * config.doppler_resolution
    
    print(f"File: {filepath.split('/')[-1]} -> Peak Velocity: {velocity:.2f} m/s")
    return velocity

def main():
    """
    Main function to process all datasets and plot the final graph.
    """
    # --- STEP 1: MODFIY THIS SECTION ---
    # List your dataset files and their corresponding angles here.
    # Format: (angle_in_degrees, 'path/to/your/file.npy')
    datasets = [
        (-30, './30_degree_negative/30_degree_negative1_2025-10-16_17-09-04.npy'),
        (-15, './15_degree_negative/15_degree_negative1_2025-10-16_15-33-29.npy'),
        (0, './0_degree/0_degree_positive1_2025-10-16_15-25-11.npy'),
        (15, './15_degree_positive/15_degree_positive2_2025-10-16_15-10-27.npy'),
        (30, './30_degree_positive/30_degree_positive2_2025-10-16_15-02-40.npy'),
    ]
    # ------------------------------------

    config = RadarConfig()
    results = []
    
    print("--- Starting Velocity Extraction ---")
    for angle, filepath in datasets:
        try:
            velocity = process_and_extract_velocity(filepath, config)
            if velocity is not None:
                results.append((angle, velocity))
        except FileNotFoundError:
            print(f"ERROR: File not found -> {filepath}. Skipping.")
        except Exception as e:
            print(f"An error occurred processing {filepath}: {e}")
            
    if not results:
        print("\nNo data was processed successfully. Exiting.")
        return

    # Unzip results for plotting
    angles, velocities = zip(*sorted(results)) # Sort by angle for a clean plot
    
    print("\n--- Plotting Results ---")
    
    # --- Create the Plot ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the measured data points
    ax.plot(angles, velocities, 'o', markersize=8, label='Measured Data')
    
    # --- Optional: Fit a Cosine Curve ---
    def cos_wave(x, A, B, C):
        # A=amplitude (max speed), B=phase shift, C=vertical offset
        return A * np.cos(np.deg2rad(x) - B) + C
    
    try:
        # Fit the curve to the data
        params, _ = curve_fit(cos_wave, angles, velocities, p0=[max(velocities), 0, 0])
        fit_angles = np.linspace(min(angles), max(angles), 200)
        fit_velocities = cos_wave(fit_angles, *params)
        ax.plot(fit_angles, fit_velocities, '--', label=f'Cosine Fit (True Speed ≈ {abs(params[0]):.2f} m/s)')
    except RuntimeError:
        print("Could not fit a cosine curve to the data.")
        
    # --- Final Plot Formatting ---
    ax.set_xlabel('Azimuth Angle (Degrees)', fontsize=12)
    ax.set_ylabel('Radial Velocity (m/s)', fontsize=12)
    ax.set_title('Radial Velocity vs. Beamsteering Angle', fontsize=14, weight='bold')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.7, alpha=0.5) # Zero velocity line
    ax.legend()
    ax.grid(True)
    
    plt.show()

if __name__ == "__main__":
    main()