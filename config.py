# config.py
data_path = 'CHI_Study_000/000000_210603-191049/ACC.csv'

plot_settings = {
    # vis for raw data
    "time_series": False,
    # useful vis functions
    "three_d": False,
    "frequency_domain": False,
    "histogram": False,
    "spectral_density": False,
    # denoising
    "fft_denoising": False,
    "moving_average": False,
    "wavelet_denoising": False,
    "bwf": False,
    # segmentation
    "segmentation": True,
    "segmentation_with_moving_average": True,
    # oscillation
    "oscillation_segmentation": True,
}

# downsampling for raw data
downsampling_rate = 1

# noise processing settings
lowpass_settings = {
    "cutoff": 1,
    "order": 5
}

moving_average_settings = {
    "window_size": 10
}

wavelet_denoising_settings = {
    "threshold": 0.2,
    "wavelet": 'db8',
    "level": 6
}

# threshold for significant change
activity_threshold = 0.6
activity_threshold_with_moving_average = 0.05

# oscillation
variability_threshold = 0.2
