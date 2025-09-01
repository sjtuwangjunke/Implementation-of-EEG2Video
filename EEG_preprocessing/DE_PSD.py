import numpy as np
import math
from scipy.fftpack import fft


def DE_PSD(data, fre, time_window):
    """
    Compute Differential Entropy (DE) and Power Spectral Density (PSD) 
    for multi-channel EEG data.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_samples)
        EEG data where each row represents one electrode and each
        column represents one time sample.
    fre : int
        Original sampling rate of the EEG data (samples per second).
    time_window : float
        Length of the sliding window (in seconds) used for FFT.

    Returns
    -------
    de : np.ndarray, shape (n_channels, n_bands)
        Differential entropy for each channel and each frequency band.
    psd : np.ndarray, shape (n_channels, n_bands)
        Power spectral density for each channel and each frequency band.
    """

    # ------------------------------------------------------------------
    # 1. Configuration of frequency bands and FFT parameters
    # ------------------------------------------------------------------
    STFTN = 200                  # FFT length (frequency-domain sampling points)
    fStart = [1, 4, 8, 14, 31]   # Start frequencies (Hz) of each band
    fEnd   = [4, 8, 14, 31, 99]  # End frequencies (Hz) of each band
    window = time_window         # Window length in seconds
    fs = fre                     # Sampling rate (Hz)

    # Convert frequency boundaries to FFT bin indices
    fStartNum = np.zeros(len(fStart), dtype=int)
    fEndNum   = np.zeros(len(fEnd), dtype=int)
    for i in range(len(fStart)):
        fStartNum[i] = int(fStart[i] / fs * STFTN)
        fEndNum[i]   = int(fEnd[i]   / fs * STFTN)

    # Number of electrodes and samples
    n_channels = data.shape[0]
    n_samples  = data.shape[1]

    # Output arrays
    psd = np.zeros([n_channels, len(fStart)])
    de  = np.zeros([n_channels, len(fStart)])

    # ------------------------------------------------------------------
    # 2. Construct Hanning window for spectral leakage reduction
    # ------------------------------------------------------------------
    Hlength = int(window * fs)  # Window length in samples
    # Hanning window formula: 0.5 - 0.5 * cos(2*pi*n/(N+1))
    Hwindow = np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (Hlength + 1))
                        for n in range(1, Hlength + 1)])

    # ------------------------------------------------------------------
    # 3. Loop over each channel and compute PSD & DE
    # ------------------------------------------------------------------
    for ch in range(n_channels):
        # Extract signal for current channel
        signal = data[ch, :]

        # Apply Hanning window to reduce spectral leakage
        windowed_signal = signal[:Hlength] * Hwindow

        # Compute FFT of the windowed segment
        fft_data = fft(windowed_signal, STFTN)

        # Compute magnitude (one-sided spectrum)
        mag_fft = np.abs(fft_data[:STFTN // 2])

        # ------------------------------------------------------------------
        # 4. Compute PSD and DE for each frequency band
        # ------------------------------------------------------------------
        for band_idx in range(len(fStart)):
            # Sum power within the band
            power_sum = 0.0
            for bin_idx in range(fStartNum[band_idx], fEndNum[band_idx] + 1):
                power_sum += mag_fft[bin_idx] ** 2

            # Average power within the band
            band_width = fEndNum[band_idx] - fStartNum[band_idx] + 1
            psd[ch, band_idx] = power_sum / band_width

            # Differential entropy: log2(100 * average power)
            de[ch, band_idx] = math.log(100 * psd[ch, band_idx], 2)

    return de, psd
