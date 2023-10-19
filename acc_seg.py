import numpy as np
from scipy.signal import find_peaks
from config import *

def detect_oscillations(data):
    # Finding peaks (maxima)
    peaks, _ = find_peaks(data)
    # Finding valleys (minima)
    valleys, _ = find_peaks(-data)
    
    # Combining and sorting all turning points
    turning_points = np.sort(np.concatenate((peaks, valleys)))
    
    oscillation_periods = []
    start_oscillation = None
    
    # Sliding window and compute variability measure
    for i in range(len(turning_points) - 1):
        start, end = turning_points[i], turning_points[i+1]
        window_data = data[start:end]
        
        if np.std(window_data) > variability_threshold:
            if start_oscillation is None:  # start of an oscillation
                start_oscillation = start
        else:
            if start_oscillation is not None:  # end of oscillation
                oscillation_periods.append((start_oscillation, end))
                start_oscillation = None
                
    return oscillation_periods
