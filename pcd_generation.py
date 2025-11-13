import sys
import os
import struct
import time
import numpy as np
import array as arr
import configuration as cfg
from scipy.ndimage import convolve1d, gaussian_filter
import matplotlib.pyplot as plt


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
    med_energy = np.median(pointCloud[:, 4])
    return pointCloud[pointCloud[:, 4] > med_energy]


if __name__ == '__main__':
    npy_filename = "./30_degree_positive2_2025-10-16_15-02-40.npy"   
    data = np.load(npy_filename, allow_pickle=True)
    print("Loaded .npy file:", npy_filename)
    print("Data shape:", data.shape)

    pointCloudProcessCFG = PointCloudProcessCFG()
    frameConfig = pointCloudProcessCFG.frameConfig
    total_frames = data.shape[0]

    all_xy_points = []
    prev_range_fft = None

    for frame_no in range(total_frames):
        frame_data = data[frame_no]
        np_frame = frame_data if np.iscomplexobj(frame_data) else bin2np_frame(frame_data)
        reshapedFrame = frameReshape(np_frame, frameConfig)
        rangeResult = rangeFFT(reshapedFrame, frameConfig)

        rangeResult = apply_gaussian_filter(rangeResult, sigma=1.0)
        rangeResult = mod_filter(rangeResult, prev_range_fft, alpha=0.6)
        prev_range_fft = rangeResult

        if pointCloudProcessCFG.enableStaticClutterRemoval:
            rangeResult = clutter_removal(rangeResult, axis=2)

        dopplerResult = dopplerFFT(rangeResult, frameConfig)
        pointCloud = frame2pointcloud(dopplerResult, pointCloudProcessCFG)

        if pointCloud.shape[0] > 0:
            all_xy_points.append(pointCloud[:, [0, 1]])  # store X, Y

        print(f"Frame {frame_no+1}/{total_frames}: {pointCloud.shape}")

    # Concatenate all XY coordinates
    if all_xy_points:
        all_xy_points = np.concatenate(all_xy_points, axis=0)
        np.save("xy_pointcloud_30deg.npy", all_xy_points)
        print("Saved XY coordinates to xy_pointcloud_0deg.npy")

        # 2D Top-Down View
        plt.figure(figsize=(7, 6))
        plt.scatter(all_xy_points[:, 0], all_xy_points[:, 1], s=8, c='royalblue', alpha=0.6)
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title("2D Global Point Cloud (Top-Down View, 0° Beam)")
        plt.grid(True)
        plt.axis('equal')
        plt.show()
    else:
        print("No points detected.")
