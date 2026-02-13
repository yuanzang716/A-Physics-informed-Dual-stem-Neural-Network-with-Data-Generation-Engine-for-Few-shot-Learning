"""
fft_analysis.py

This script provides a standalone, open-source-ready implementation for calculating
the initial coarse estimate of a filament's diameter from its Fraunhofer
diffraction pattern, based on a 1D Fast Fourier Transform (FFT) analysis.

It replicates the core logic of the initial estimation stage described in the
accompanying manuscript, including:
  1.  Image preprocessing (cropping, projection to 1D).
  2.  FFT-based frequency peak detection (both discrete and with sub-pixel
      refinement via parabolic interpolation).
  3.  Calculation of the diameter based on physical constants.

This script is intended to be run from the command line to analyze directories
of diffraction images and produce summary statistics, demonstrating the inherent
limitations of FFT-based estimation and motivating the more advanced methods
presented in the main paper.
"""

import os
import glob
import argparse
import cv2
import numpy as np
import pandas as pd


def parabolic_interpolation(y_m1: float, y0: float, y_p1: float) -> float:
    """Performs 3-point parabolic interpolation to find a sub-pixel peak location.

    Args:
        y_m1, y0, y_p1: Amplitudes of the spectrum at k-1, k, and k+1.

    Returns:
        The fractional offset delta (-0.5 to 0.5) from the center point k.
    """
    # To avoid division by zero if the three points are collinear
    denominator = (y_m1 - 2 * y0 + y_p1)
    if np.isclose(denominator, 0):
        return 0.0
    return 0.5 * (y_m1 - y_p1) / denominator


def fft_initializer_from_image(
    image_path: str,
    wavelength_um: float,
    focal_length_um: float,
    pixel_size_um: float,
    crop_pixels: int = 2
) -> dict:
    """Calculates a coarse diameter estimate from a single diffraction image using FFT.

    Args:
        image_path: Path to the BMP image file.
        wavelength_um: Laser wavelength in micrometers.
        focal_length_um: Lens focal length in micrometers.
        pixel_size_um: CCD pixel size in micrometers.
        crop_pixels: Number of pixels to crop from each border.

    Returns:
        A dictionary containing intermediate and final calculation results.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")

    # 1. Preprocessing (mimicking MATLAB script)
    if crop_pixels > 0:
        img = img[crop_pixels:-crop_pixels, crop_pixels:-crop_pixels]
    img = img.astype(np.float64)

    # Project to 1D by taking the max intensity along each row (max over columns)
    signal_1d = img.max(axis=1)

    # 2. FFT Analysis
    signal_length = len(signal_1d)
    spatial_freq_sampling = 1.0 / pixel_size_um

    # Compute one-sided power spectrum
    fft_coeffs = np.fft.fft(signal_1d)
    power_spectrum = np.abs(fft_coeffs / signal_length)[:signal_length // 2 + 1]
    if len(power_spectrum) > 2:
        power_spectrum[1:-1] *= 2
    
    freq_axis = spatial_freq_sampling / signal_length * np.arange(signal_length // 2 + 1)

    # 3. Peak Finding (Discrete and Interpolated)
    # Find the dominant frequency peak, excluding the DC component (k=0)
    peak_index_k = int(np.argmax(power_spectrum[1:]) + 1)
    dominant_freq_discrete = float(freq_axis[peak_index_k])

    # Sub-pixel refinement with parabolic interpolation
    if 1 <= peak_index_k < len(power_spectrum) - 1:
        delta_k = parabolic_interpolation(
            power_spectrum[peak_index_k - 1],
            power_spectrum[peak_index_k],
            power_spectrum[peak_index_k + 1]
        )
    else:
        delta_k = 0.0
    
    peak_index_interp = float(peak_index_k) + delta_k
    dominant_freq_interp = (spatial_freq_sampling / signal_length) * peak_index_interp

    # 4. Diameter Calculation
    diameter_discrete = dominant_freq_discrete * wavelength_um * focal_length_um
    diameter_interp = dominant_freq_interp * wavelength_um * focal_length_um

    return {
        'filename': os.path.basename(image_path),
        'peak_index_k': peak_index_k,
        'dominant_freq_discrete': dominant_freq_discrete,
        'diameter_discrete': diameter_discrete,
        'peak_index_interp': peak_index_interp,
        'dominant_freq_interp': dominant_freq_interp,
        'diameter_interp': diameter_interp,
    }


def analyze_directory(directory: str, wavelength_um: float, focal_length_um: float, pixel_size_um: float, ground_truth_um: float) -> pd.DataFrame:
    """Analyzes all BMP images in a directory and returns a summary DataFrame."""
    image_paths = sorted(glob.glob(os.path.join(directory, '*.BMP')))
    if not image_paths:
        print(f"Warning: No BMP images found in {directory}")
        return pd.DataFrame()

    results = []
    for path in image_paths:
        try:
            res = fft_initializer_from_image(path, wavelength_um, focal_length_um, pixel_size_um)
            results.append(res)
        except FileNotFoundError as e:
            print(e)
            continue
    
    df = pd.DataFrame(results)
    df['rel_error_discrete_%'] = 100 * (df['diameter_discrete'] - ground_truth_um) / ground_truth_um
    df['rel_error_interp_%'] = 100 * (df['diameter_interp'] - ground_truth_um) / ground_truth_um
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FFT-based Coarse Diameter Estimation Analysis")
    parser.add_argument('--wavelength', type=float, default=0.78, help="Wavelength in micrometers")
    parser.add_argument('--pixel_size', type=float, default=4.8, help="Pixel size in micrometers")
    parser.add_argument('--ground_truth', type=float, default=100.2, help="Ground truth diameter in micrometers")
    args = parser.parse_args()

    _DEFAULT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    
    datasets = {
        '75mm': {
            'path': os.path.join(_DEFAULT_DATA_DIR, 'Diameter_100.2_75mm', 'train_real'),
            'focal_length': 75000.0
        },
        '120mm': {
            'path': os.path.join(_DEFAULT_DATA_DIR, 'Diameter_100.2_120mm', 'train_real'),
            'focal_length': 120000.0
        }
    }

    all_results = []

    print("--- FFT Initializer Analysis ---")
    for name, params in datasets.items():
        print(f"\nAnalyzing {name} dataset...")
        df = analyze_directory(
            directory=params['path'],
            wavelength_um=args.wavelength,
            focal_length_um=params['focal_length'],
            pixel_size_um=args.pixel_size,
            ground_truth_um=args.ground_truth
        )
        if df.empty:
            continue
        
        df['focal_length'] = name
        all_results.append(df)

        summary = {
            'method': ['FFT (discrete peak)', 'FFT (interpolated peak)'],
            'mean_diameter_um': [
                df['diameter_discrete'].mean(),
                df['diameter_interp'].mean()
            ],
            'std_diameter_um': [
                df['diameter_discrete'].std(),
                df['diameter_interp'].std()
            ],
            'mean_rel_error_%': [
                df['rel_error_discrete_%'].mean(),
                df['rel_error_interp_%'].mean()
            ],
            'std_rel_error_%': [
                df['rel_error_discrete_%'].std(),
                df['rel_error_interp_%'].std()
            ]
        }
        summary_df = pd.DataFrame(summary)
        print(f"Summary for {name}:")
        print(summary_df.to_string(index=False))

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        output_path = 'fft_analysis_results.csv'
        final_df.to_csv(output_path, index=False)
        print(f"\nFull results saved to: {output_path}")

