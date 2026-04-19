"""
evaluation.py
Objective metrics for audio quality assessment.
    - SNR  : Signal-to-Noise Ratio
    - LSD  : Log-Spectral Distance
    - ΔLSD : improvement from low-bitrate to reconstructed
"""

import numpy as np


def _snr(reference: np.ndarray, degraded: np.ndarray) -> float:
    """SNR in dB. Higher is better."""
    n = min(len(reference), len(degraded))
    ref, deg = reference[:n], degraded[:n]
    noise = ref - deg
    signal_power = np.mean(ref ** 2)
    noise_power  = np.mean(noise ** 2)
    if noise_power < 1e-12:
        return 99.9
    return 10 * np.log10(signal_power / noise_power)


def _log_spectral_distance(reference: np.ndarray, degraded: np.ndarray, sr: int) -> float:
    """
    LSD (dB). Lower is better.
    Computed as mean over frames of RMS log-spectral distortion.
    """
    n_fft = 1024
    n = min(len(reference), len(degraded))
    ref, deg = reference[:n], degraded[:n]

    # Pad to multiple of n_fft
    pad = (-n) % n_fft
    ref = np.pad(ref, (0, pad))
    deg = np.pad(deg, (0, pad))

    frames_ref = ref.reshape(-1, n_fft)
    frames_deg = deg.reshape(-1, n_fft)

    lsd_per_frame = []
    for r, d in zip(frames_ref, frames_deg):
        R = np.abs(np.fft.rfft(r * np.hanning(n_fft))) + 1e-8
        D = np.abs(np.fft.rfft(d * np.hanning(n_fft))) + 1e-8
        log_diff = 20 * np.log10(R) - 20 * np.log10(D)
        lsd_per_frame.append(np.sqrt(np.mean(log_diff ** 2)))

    return float(np.mean(lsd_per_frame))


def compute_metrics(
    y_orig: np.ndarray,
    y_low:  np.ndarray,
    y_recon: np.ndarray,
    sr: int,
) -> dict:
    """Return a dict of all evaluation metrics."""
    snr_low   = _snr(y_orig, y_low)
    snr_recon = _snr(y_orig, y_recon)
    lsd_low   = _log_spectral_distance(y_orig, y_low,   sr)
    lsd_recon = _log_spectral_distance(y_orig, y_recon, sr)

    return {
        "snr_low":    snr_low,
        "snr_recon":  snr_recon,
        "snr_delta":  snr_recon - snr_low,
        "lsd_low":    lsd_low,
        "lsd_recon":  lsd_recon,
        "lsd_delta":  lsd_low - lsd_recon,   # positive = improvement
    }
