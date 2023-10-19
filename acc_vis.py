import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import welch
from config import *
from acc_noise import *
from acc_seg import segment_data


# Load data
acc = pd.read_csv(data_path, header=None)
start_time = acc.iloc[0][0]
frequency = acc.iloc[1][0]
data = acc.drop([0,1])
data = data.apply(lambda x:x/64, axis=0)
data.columns = ['x', 'y', 'z']
data['timestamps'] = np.arange(data.shape[0])/frequency

# Downsample data
data = data[::downsampling_rate] 

# vis for raw data
def plot_time_series():
    fig, axs = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
    axs[0].plot(data['timestamps'], data['x'], label='X-axis')
    axs[1].plot(data['timestamps'], data['y'], label='Y-axis')
    axs[2].plot(data['timestamps'], data['z'], label='Z-axis')

    axs[0].set_ylabel('Acceleration (1/64 g)')
    axs[1].set_ylabel('Acceleration (1/64 g)')
    axs[2].set_ylabel('Acceleration (1/64 g)')

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    axs[2].set_xlabel('Time (seconds)')

    plt.suptitle('3-axis Accelerometer Data')
    plt.grid(True)
    plt.draw()
    plt.pause(0.1)

# useful vis functions
def plot_three_d():
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    t = np.linspace(0, 1, len(data))
    sc = ax.scatter(data['x'], data['y'], data['z'], c=t, cmap='viridis', s=1)
    ax.set_title('3D Path of Accelerometer Data')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.colorbar(sc, label='Normalized Time')
    plt.draw()
    plt.pause(0.1)

def plot_frequency_domain():
    plt.figure(figsize=(10, 6))
    fft_x = np.fft.fft(data['x'])
    fft_y = np.fft.fft(data['y'])
    fft_z = np.fft.fft(data['z'])
    freq = np.fft.fftfreq(len(data), 1/frequency)
    plt.plot(freq, np.abs(fft_x), label='X-axis')
    plt.plot(freq, np.abs(fft_y), label='Y-axis')
    plt.plot(freq, np.abs(fft_z), label='Z-axis')
    plt.title('Frequency Domain Representation')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)
    plt.draw()
    plt.pause(0.1)

def plot_histogram():
    plt.figure(figsize=(10, 6))
    plt.hist(data['x'], bins=50, alpha=0.5, label='X-axis')
    plt.hist(data['y'], bins=50, alpha=0.5, label='Y-axis')
    plt.hist(data['z'], bins=50, alpha=0.5, label='Z-axis')
    plt.title('Histogram of Accelerometer Data')
    plt.xlabel('Acceleration (1/64 g)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.draw()
    plt.pause(0.1)

def plot_spectral_density():
    plt.figure(figsize=(10, 6))
    f, Pxx_x = welch(data['x'], fs=frequency)
    f, Pxx_y = welch(data['y'], fs=frequency)
    f, Pxx_z = welch(data['z'], fs=frequency)
    plt.semilogy(f, Pxx_x, label='X-axis')
    plt.semilogy(f, Pxx_y, label='Y-axis')
    plt.semilogy(f, Pxx_z, label='Z-axis')
    plt.title('Power Spectral Density')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.legend()
    plt.grid(True)
    plt.draw()
    plt.pause(0.1)

# denoising
def fft(col, n_components):
    n = len(col)
    # compute the fft
    fft_vals = np.fft.fft(col, n)
    # compute power spectrum density
    # squared magnitude of each fft coefficient
    PSD = fft_vals * np.conj(fft_vals) / n
    # keep high frequencies
    _mask = PSD > n_components
    fft_vals = _mask * fft_vals
    # inverse Fourier transform
    clean_data = np.fft.ifft(fft_vals)
    clean_data = clean_data.real
    return clean_data

def plot_fft_denoising():
    plt.figure(figsize=(10, 6))
    np_x = data['x'].to_numpy()
    np_y = data['y'].to_numpy()
    np_z = data['z'].to_numpy()
    np_ts = data['timestamps'].to_numpy()
    plt.plot(np_ts, fft(np_y, 75))
    plt.plot(np_ts, fft(np_x, 75))
    plt.plot(np_ts, fft(np_z, 75))
    plt.legend(["x", "y", 'z'], loc="lower right")
    plt.title('FFT Denoising')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceleration (1/64 g)')
    plt.grid(True)
    plt.draw()
    plt.pause(0.1)

def plot_moving_average():
    fig, axs = plt.subplots(3, 1, figsize=(16, 8), sharex=True) 
    smoothed_x = moving_average(data['x'], **moving_average_settings)
    smoothed_y = moving_average(data['y'], **moving_average_settings)
    smoothed_z = moving_average(data['z'], **moving_average_settings)
    
    axs[0].plot(data['timestamps'], smoothed_x, label='Smoothed X-axis')
    axs[1].plot(data['timestamps'], smoothed_y, label='Smoothed Y-axis')
    axs[2].plot(data['timestamps'], smoothed_z, label='Smoothed Z-axis')
    
    axs[0].set_ylabel('Acceleration (1/64 g)')
    axs[1].set_ylabel('Acceleration (1/64 g)')
    axs[2].set_ylabel('Acceleration (1/64 g)')
    
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    
    axs[2].set_xlabel('Time (seconds)')
    
    plt.suptitle('Moving Average Smoothed Data')
    plt.grid(True)
    plt.draw()
    plt.pause(0.1)

def plot_wavelet_denoising():
    fig, axs = plt.subplots(3, 1, figsize=(16, 8), sharex=True) 
    denoised_x = wavelet_denoising(data['x'], **wavelet_denoising_settings)
    denoised_y = wavelet_denoising(data['y'], **wavelet_denoising_settings)
    denoised_z = wavelet_denoising(data['z'], **wavelet_denoising_settings)
    
    axs[0].plot(data['timestamps'], denoised_x, label='Denoised X-axis')
    axs[1].plot(data['timestamps'], denoised_y, label='Denoised Y-axis')
    axs[2].plot(data['timestamps'], denoised_z, label='Denoised Z-axis')
    
    axs[0].set_ylabel('Acceleration (1/64 g)')
    axs[1].set_ylabel('Acceleration (1/64 g)')
    axs[2].set_ylabel('Acceleration (1/64 g)')
    
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    
    axs[2].set_xlabel('Time (seconds)') 
    
    plt.suptitle('Wavelet Denoised Data')
    plt.grid(True)
    plt.draw()
    plt.pause(0.1)

def plot_bwf():
    fig, axs = plt.subplots(3, 1, figsize=(16, 8), sharex=True) 
    np_x = data['x'].to_numpy()
    np_y = data['y'].to_numpy()
    np_z = data['z'].to_numpy()
    np_ts = data['timestamps'].to_numpy()
    frequency = 1 / (np_ts[1] - np_ts[0])
    
    axs[0].plot(np_ts, bwf(np_x, frequency), label='Filtered X')
    axs[1].plot(np_ts, bwf(np_y, frequency), label='Filtered Y')
    axs[2].plot(np_ts, bwf(np_z, frequency), label='Filtered Z')
    
    axs[0].set_ylabel('Acceleration (1/64 g)')
    axs[1].set_ylabel('Acceleration (1/64 g)')
    axs[2].set_ylabel('Acceleration (1/64 g)')
    
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    
    axs[2].set_xlabel('Time (seconds)')
    
    plt.suptitle('Butterworth Filtered Data')
    plt.grid(True)
    plt.draw()
    plt.pause(0.1)

# segmentation
def plot_segments():
    fig, axs = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
    
    # Plotting the XYZ data
    axs[0].plot(data['timestamps'], data['x'], label='X-axis')
    axs[1].plot(data['timestamps'], data['y'], label='Y-axis')
    axs[2].plot(data['timestamps'], data['z'], label='Z-axis')
    
    # Identify significant changes in XYZ data and mark the segments
    threshold = activity_threshold  # This should be imported from your config file
    for i in range(1, len(data)):
        if abs(data['x'].iloc[i] - data['x'].iloc[i-1]) > threshold:
            axs[0].axvline(x=data['timestamps'].iloc[i], color='r', linestyle='--')
        if abs(data['y'].iloc[i] - data['y'].iloc[i-1]) > threshold:
            axs[1].axvline(x=data['timestamps'].iloc[i], color='r', linestyle='--')
        if abs(data['z'].iloc[i] - data['z'].iloc[i-1]) > threshold:
            axs[2].axvline(x=data['timestamps'].iloc[i], color='r', linestyle='--')
    
    axs[0].set_ylabel('Acceleration (1/64 g)')
    axs[1].set_ylabel('Acceleration (1/64 g)')
    axs[2].set_ylabel('Acceleration (1/64 g)')

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    axs[2].set_xlabel('Time (seconds)')

    plt.suptitle('3-axis Accelerometer Data with Activity Segments')
    plt.grid(True)
    plt.draw()
    plt.pause(0.1)

def plot_segments_with_moving_average():
    fig, axs = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
    
    data['x_processed'] = moving_average(data['x'], **moving_average_settings)
    data['y_processed'] = moving_average(data['y'], **moving_average_settings)
    data['z_processed'] = moving_average(data['z'], **moving_average_settings)

    # Plotting the XYZ data
    axs[0].plot(data['timestamps'], data['x_processed'], label='X-axis Processed')
    axs[1].plot(data['timestamps'], data['y_processed'], label='Y-axis Processed')
    axs[2].plot(data['timestamps'], data['z_processed'], label='Z-axis Processed')
    
    # Identify significant changes in XYZ data and mark the segments
    threshold = activity_threshold  # This should be imported from your config file
    for i in range(1, len(data)):
        if abs(data['x_processed'].iloc[i] - data['x_processed'].iloc[i-1]) > activity_threshold:
            axs[0].axvline(x=data['timestamps'].iloc[i], color='r', linestyle='--')
        if abs(data['y_processed'].iloc[i] - data['y_processed'].iloc[i-1]) > activity_threshold:
            axs[1].axvline(x=data['timestamps'].iloc[i], color='r', linestyle='--')
        if abs(data['z_processed'].iloc[i] - data['z_processed'].iloc[i-1]) > activity_threshold:
            axs[2].axvline(x=data['timestamps'].iloc[i], color='r', linestyle='--')
    
    axs[0].set_ylabel('Acceleration (1/64 g)')
    axs[1].set_ylabel('Acceleration (1/64 g)')
    axs[2].set_ylabel('Acceleration (1/64 g)')

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    axs[2].set_xlabel('Time (seconds)')

    plt.suptitle('3-axis Accelerometer Data with Activity Segments [moving_average]')
    plt.grid(True)
    plt.draw()
    plt.pause(0.1)


if __name__ == "__main__":
    # Check configuration and generate requested plots
    if plot_settings.get("time_series", False):
        plot_time_series()
    if plot_settings.get("three_d", False):
        plot_three_d()
    if plot_settings.get("frequency_domain", False):
        plot_frequency_domain()
    if plot_settings.get("histogram", False):
        plot_histogram()
    if plot_settings.get("spectral_density", False):
        plot_spectral_density()
    if plot_settings.get("fft_denoising", False):
        plot_fft_denoising()
    if plot_settings.get("moving_average", False):
        plot_moving_average()
    if plot_settings.get("wavelet_denoising", False):
        plot_wavelet_denoising()
    if plot_settings.get("bwf", False):
        plot_bwf()
    if plot_settings.get("segmentation", False):
        plot_segments()
    if plot_settings.get("segmentation_with_moving_average", False):
        plot_segments_with_moving_average()

    plt.pause(0.001)
    input("Press [enter] to close the plots.")
