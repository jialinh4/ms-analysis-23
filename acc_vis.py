import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import welch
from config import plot_settings, data_path

# Load data
acc = pd.read_csv(data_path, header=None)
start_time = acc.iloc[0][0]
frequency = acc.iloc[1][0]
data = acc.drop([0,1])
data = data.apply(lambda x:x/64, axis=0)
data.columns = ['x', 'y', 'z']
data['timestamps'] = np.arange(data.shape[0])/frequency

def plot_time_series():
    plt.figure(figsize=(10, 6))
    plt.plot(data['timestamps'], data['x'], label='X-axis')
    plt.plot(data['timestamps'], data['y'], label='Y-axis')
    plt.plot(data['timestamps'], data['z'], label='Z-axis')
    plt.title('3-axis Accelerometer Data')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceleration (1/64 g)')
    plt.legend()
    plt.grid(True)
    plt.draw()
    plt.pause(0.1)

def plot_three_d():
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(data['x'], data['y'], [0]*len(data), color='red', label='X-axis')  # Red color for X-axis
    ax.plot(data['x'], [0]*len(data), data['z'], color='green', label='Y-axis')  # Green color for Y-axis
    ax.plot([0]*len(data), data['y'], data['z'], color='blue', label='Z-axis')  # Blue color for Z-axis
    ax.set_title('3D Path of Accelerometer Data')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.legend()
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
    plt.show()
