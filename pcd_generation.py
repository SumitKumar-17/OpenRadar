
import sys
import os
import struct
import time
import numpy as np
import array as arr
import configuration as cfg  # Assuming you have this file
from scipy.ndimage import convolve1d, gaussian_filter
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN  # NEW: Import DBSCAN for filtering

def read8byte(x):
    return struct.unpack('<hhhh', x)


class FrameConfig:
    def __init__(self):
        self.numTxAntennas = cfg.NUM_TX
        self.numRxAntennas = cfg.NUM_RX
        self.numLoopsPerFrame = cfg.LOOPS_PER_FRAME
        self.numADCSamples = cfg.ADC_SAMPLES
        self.numAngleBins = cfg.NUM_ANGLE_BINS
        self.numChirpsPerFrame = self.numTxAntennas * self.numLoopsPerFrame
        self.numRangeBins = self.numADCSamples
        self.numDopplerBins = self.numLoopsPerFrame
        self.chirpSize = self.numRxAntennas * self.numADCSamples
        self.chirpLoopSize = self.chirpSize * self.numTxAntennas
        self.frameSize = self.chirpLoopSize * self.numLoopsPerFrame


class PointCloudProcessCFG:
    def __init__(self):
        self.frameConfig = FrameConfig()
        self.enableStaticClutterRemoval = True
        self.EnergyTop128 = True
        self.RangeCut = False
        self.outputVelocity = True
        self.outputSNR = True
        self.outputRange = True
        self.outputInMeter = True
        self.EnergyThrMed = True
        self.ConstNoPCD = False
        self.dopplerToLog = False


def bin2np_frame(bin_frame):
    np_frame = np.zeros(shape=(len(bin_frame) // 2), dtype=np.complex128)
    np_frame[0::2] = bin_frame[0::4] + 1j * bin_frame[2::4]
    np_frame[1::2] = bin_frame[1::4] + 1j * bin_frame[3::4]
    return np_frame


def frameReshape(frame, frameConfig):
    frameWithChirp = np.reshape(
        frame,
        (frameConfig.numLoopsPerFrame, frameConfig.numTxAntennas, frameConfig.numRxAntennas, -1)
    )
    return frameWithChirp.transpose(1, 2, 0, 3)


def rangeFFT(reshapedFrame, frameConfig):
    windowedBins1D = reshapedFrame
    return np.fft.fft(windowedBins1D)


def clutter_removal(input_val, axis=0):
    reordering = np.arange(len(input_val.shape))
    reordering[0] = axis
    reordering[axis] = 0
    input_val = input_val.transpose(reordering)
    mean = input_val.mean(0)
    output_val = input_val - mean
    return output_val.transpose(reordering)


def dopplerFFT(rangeResult, frameConfig):
    windowedBins2D = rangeResult * np.reshape(np.hamming(frameConfig.numLoopsPerFrame), (1, 1, -1, 1))
    dopplerFFTResult = np.fft.fft(windowedBins2D, axis=2)
    dopplerFFTResult = np.fft.fftshift(dopplerFFTResult, axes=2)
    return dopplerFFTResult


def mod_filter(current_frame, prev_frame, alpha=0.7):
    if prev_frame is None:
        return current_frame
    return alpha * prev_frame + (1 - alpha) * current_frame


def apply_gaussian_filter(frame, sigma=1.0):
    filtered = gaussian_filter(np.abs(frame), sigma=sigma)
    return filtered * np.exp(1j * np.angle(frame))


def naive_xyz(virtual_ant, num_tx=3, num_rx=4, fft_size=64):
    num_detected_obj = virtual_ant.shape[1]
    azimuth_ant = virtual_ant[:2 * num_rx, :]
    azimuth_ant_padded = np.zeros((fft_size, num_detected_obj), dtype=np.complex128)
    azimuth_ant_padded[:2 * num_rx, :] = azimuth_ant

    azimuth_fft = np.fft.fft(azimuth_ant_padded, axis=0)
    k_max = np.argmax(np.abs(azimuth_fft), axis=0)
    peak_1 = np.array([azimuth_fft[k, i] for i, k in enumerate(k_max)])

    k_max[k_max > (fft_size // 2) - 1] -= fft_size
    wx = 2 * np.pi / fft_size * k_max
    x_vector = wx / np.pi

    elevation_ant = virtual_ant[2 * num_rx:, :]
    elevation_fft = np.fft.fft(elevation_ant, axis=0)
    elevation_max = np.argmax(np.log2(np.abs(elevation_fft)), axis=0)
    peak_2 = np.array([elevation_fft[k, i] for i, k in enumerate(elevation_max)])

    wz = np.angle(peak_1 * peak_2.conj() * np.exp(1j * 2 * wx))
    z_vector = wz / np.pi
    y_vector = np.sqrt(np.maximum(1 - x_vector**2 - z_vector**2, 0))
    return x_vector, y_vector, z_vector


def frame2pointcloud(dopplerResult, pointCloudProcessCFG):
    dopplerResultSumAllAntenna = np.sum(dopplerResult, axis=(0, 1))
    dopplerResultInDB = np.abs(dopplerResultSumAllAntenna)

    cfarResult = np.zeros(dopplerResultInDB.shape, bool)
    if pointCloudProcessCFG.EnergyTop128:
        top_size = 128
        energyThre128 = np.partition(dopplerResultInDB.ravel(), dopplerResultInDB.size - top_size - 1)[
            dopplerResultInDB.size - top_size - 1]
        cfarResult[dopplerResultInDB > energyThre128] = True

    det_peaks_indices = np.argwhere(cfarResult)
    if len(det_peaks_indices) == 0:
        return np.empty((0, 6))

    R = det_peaks_indices[:, 1].astype(np.float64)
    V = (det_peaks_indices[:, 0] - FrameConfig().numDopplerBins // 2).astype(np.float64)
    if pointCloudProcessCFG.outputInMeter:
        R *= cfg.RANGE_RESOLUTION
        V *= cfg.DOPPLER_RESOLUTION
    energy = dopplerResultInDB[cfarResult]

    AOAInput = dopplerResult[:, :, cfarResult].reshape(12, -1)
    if AOAInput.shape[1] == 0:
        return np.empty((0, 6))

    x_vec, y_vec, z_vec = naive_xyz(AOAInput)
    x, y, z = x_vec * R, y_vec * R, z_vec * R

    pointCloud = np.vstack((x, y, z, V, energy, R)).T
    
    # 1. First, filter by energy (as before)
    med_energy = np.median(pointCloud[:, 4])
    pc_filtered_energy = pointCloud[pointCloud[:, 4] > med_energy]
    
    # 2. NEW: Filter by distance (Range)
    #    Keep only points where the range (column 5) is between 4m and 6m
    if pc_filtered_energy.shape[0] > 0:
        range_values = pc_filtered_energy[:, 5]
        range_mask = (range_values >= 4) & (range_values <= 6)
        pc_filtered_range = pc_filtered_energy[range_mask]
        return pc_filtered_range
    else:
        return np.empty((0, 6))

def process_radar_file(npy_filename, pointCloudProcessCFG, angle_deg):
    """
    Loads and processes a single .npy radar data file.
    Applies coordinate transformation based on the steering angle.
    Returns a (N, 2) numpy array of (X, Y) coordinates.
    """
    print(f"Loading and processing {npy_filename} for angle {angle_deg}°...")
    try:
        data = np.load(npy_filename, allow_pickle=True)
        print(f"Loaded {npy_filename}, shape: {data.shape}")
    except FileNotFoundError:
        print(f"Warning: File not found {npy_filename}. Skipping.")
        return np.empty((0, 2))

    frameConfig = pointCloudProcessCFG.frameConfig
    total_frames = data.shape[0]

    all_xy_points_local = []
    prev_range_fft = None

    for frame_no in range(total_frames):
        frame_data = data[frame_no]
        np_frame = frame_data if np.iscomplexobj(frame_data) else bin2np_frame(frame_data)
        reshapedFrame = frameReshape(np_frame, frameConfig)
        rangeResult = rangeFFT(reshapedFrame, frameConfig)

        # --- YAHAN PAR CHANGE KIYA GAYA HAI ---
        # Gaussian filter ko strong kar diya gaya hai.
        # Pehle sigma=1.0 tha, ab hum 2.0 try kar rahe hain.
        # Aap is value ko aur badha (e.g., 2.5) ya ghata (e.g., 1.5) sakte hain.
        rangeResult = apply_gaussian_filter(rangeResult, sigma=2.0) # <-- TUNED PARAMETER
        
        rangeResult = mod_filter(rangeResult, prev_range_fft, alpha=0.6)
        prev_range_fft = rangeResult

        if pointCloudProcessCFG.enableStaticClutterRemoval:
            rangeResult = clutter_removal(rangeResult, axis=2)

        dopplerResult = dopplerFFT(rangeResult, frameConfig)
        pointCloud = frame2pointcloud(dopplerResult, pointCloudProcessCFG)

        if pointCloud.shape[0] > 0:
            all_xy_points_local.append(pointCloud[:, [0, 1]])  # store Local X, Y

    if not all_xy_points_local:
        print(f"No points detected in {npy_filename}.")
        return np.empty((0, 2))

    xy_local = np.concatenate(all_xy_points_local, axis=0)

    # Apply 2D rotation matrix for coordinate transformation
    print(f"Applying {angle_deg}° rotation to {xy_local.shape[0]} points...")
    angle_rad = np.deg2rad(angle_deg)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    
    x_local = xy_local[:, 0]
    y_local = xy_local[:, 1]
    
    x_global = x_local * cos_theta - y_local * sin_theta
    y_global = x_local * sin_theta + y_local * cos_theta
    
    xy_global = np.vstack((x_global, y_global)).T
    
    print(f"Finished processing {npy_filename}.")
    return xy_global

if __name__ == '__main__':
    # 1. Define the files and their corresponding angles
    #    !!! YOU MUST UPDATE THESE PATHS to your .npy files !!!
    base_path = r"./data"  # Example base path, update as needed
    

    files_to_process = {
        0: os.path.join(base_path, "0 degree", "0_degree_positive1_2025-10-16_15-25-11.npy"),
        -15: os.path.join(base_path, "15 degree negative", "15_degree_negative1_2025-10-16_15-33-29.npy"),
         15:  os.path.join(base_path, "15 degree positive", "15_degree_positive1_2025-10-16_15-09-48.npy"),
        30:  os.path.join(base_path, "30 degree positive", "30_degree_positive1_2025-10-16_15-01-59.npy"),
        -30: os.path.join(base_path, "30 degree negative", "30_degree_negative1_2025-10-16_17-09-04.npy")
    }

    # 2. Define colors for each angle
    angle_colors = {
        -30: 'purple',
        -15: 'blue',
         0:  'green',
        15:  'orange',
        30:  'red'
    }

    # 3. Initialize common configuration
    pointCloudProcessCFG = PointCloudProcessCFG()
    
    # 4. Process each file and store results for plotting
    plot_data = []
    
    print("Starting point cloud processing for all angles...")
    for angle, npy_filename in files_to_process.items():
        # Pass the angle to the processing function
        xy_points_global = process_radar_file(npy_filename, pointCloudProcessCFG, angle)
        
        if xy_points_global.shape[0] > 0:
            
            # --- MORE AGGRESSIVE DBSCAN SETTINGS ---
            #
            # We are increasing min_samples from 10 to 30.
            # This means a cluster must be much denser to be kept.
            #
            # We are also decreasing eps from 0.5 to 0.4.
            # This means points must be closer together.
            #
            # Feel free to tune these two values!
            # To filter MORE: increase min_samples, decrease eps
            # To filter LESS: decrease min_samples, increase eps
            
            print(f"Applying MORE AGGRESSIVE DBSCAN filtering to {angle}° data...")
            db = DBSCAN(eps=0.4, min_samples=30).fit(xy_points_global) # <-- TUNED PARAMETERS
            labels = db.labels_
            
            # Keep only points that are part of a cluster (label != -1)
            filtered_points = xy_points_global[labels != -1]
            num_removed = xy_points_global.shape[0] - filtered_points.shape[0]
            print(f"DBSCAN removed {num_removed} noise points (was {xy_points_global.shape[0]} points, now {filtered_points.shape[0]}).")

            if filtered_points.shape[0] > 0:
                plot_data.append({
                    'angle': angle,
                    'points': filtered_points, # Use filtered points
                    'color': angle_colors[angle]
                })
            else:
                print(f"No non-noise points found for angle {angle}° after DBSCAN.")
    
    print("All files processed. Generating plot...")

    # 5. Plot all point clouds on one figure
    plt.figure(figsize=(12, 10))
    
    if not plot_data:
        print("No data to plot.")
    else:
        for data_dict in plot_data:
            plt.scatter(
                data_dict['points'][:, 0], 
                data_dict['points'][:, 1], 
                s=8, 
                c=data_dict['color'], 
                alpha=0.6, 
                label=f"{data_dict['angle']}° Beam"
            )

        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title("Global Point Cloud (Top-Down View, All Angles) - Rotated & Aggressively Filtered")
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        plt.show()