# config.py
data_path = 'sample_data/acc.csv'

plot_settings = {
    "time_series": True,
    "three_d": True,
    "frequency_domain": True,
    "histogram": True,
    "spectral_density": True,
    "fft_denoising": True,
    "moving_average": True,
    "wavelet_denoising": True,
    "bwf": True,
}

downsampling_rate = 1

# Noise Processing Settings
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
