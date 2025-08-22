# OpenRadar Azimuth Study

I focused on understanding how raw radar ADC data is processed into **range-azimuth heatmaps**.

## What I Learned
- How to **organize raw ADC samples** into complex I/Q format for further processing.
- Running the **range FFT** to get distance information from chirps.
- Using **clutter removal** to filter static background reflections.
- Arranging **virtual antennas (TDM MIMO)** for spatial processing.
- Applying **Capon beamforming** with steering vectors to estimate target azimuth angles.
- Converting outputs into **range-azimuth heatmaps** and animating them over time.

## Output
- Generates a time-series animation of heatmaps showing how targets move in range and azimuth.
- Results are visualized in dB scale with configurable dynamic range, colormap, and frame interval.

## Usage
```bash
python azimuth.py
