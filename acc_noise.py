import pandas as pd
from scipy.signal import butter, lfilter, filtfilt
import pywt
import math

def bwf(data, frequency):
    cutoff_freq = math.ceil(1.2*frequency) #slightly higher then base freq
    nyq = frequency/2
    order = 4 #modelling the function as a 4th order polynomial
    normal_cutoff = nyq/cutoff_freq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def moving_average(data, window_size):
    return data.rolling(window=window_size).mean()

def wavelet_denoising(data, threshold=0.2, wavelet='db8', level=6):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs_thresholded, wavelet)

def get_summed_moving_average(data, window_size):
    data['x_processed'] = moving_average(data['x'], window_size)
    data['y_processed'] = moving_average(data['y'], window_size)
    data['z_processed'] = moving_average(data['z'], window_size)
    return data['x_processed'] + data['y_processed'] + data['z_processed']
