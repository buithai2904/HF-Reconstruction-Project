"""
bwe_methods.py
Bandwidth Extension — traditional signal processing approaches.
"""

import numpy as np
from scipy import signal


# ── 1. Low-pass filter (simulate low-bitrate) ───────────────────────────────────
def apply_lowpass_filter(y: np.ndarray, cutoff_hz: float, sr: int) -> np.ndarray:
    """Apply a sharp low-pass Butterworth filter to simulate low-bitrate audio."""
    nyq = sr / 2.0
    norm_cutoff = cutoff_hz / nyq
    norm_cutoff = np.clip(norm_cutoff, 0.001, 0.999)
    b, a = signal.butter(8, norm_cutoff, btype="low", analog=False)
    y_filtered = signal.filtfilt(b, a, y)
    return y_filtered.astype(np.float32)


# ── 2. Spectral Folding BWE ──────────────────────────────────────────────────────
def reconstruct_bwe_spectral_folding(
    y_low: np.ndarray,
    sr: int,
    cutoff_hz: float,
    fold_factor: int = 2,
    hf_gain: float = 0.7,
) -> np.ndarray:
    """
    Mirror the low-frequency spectrum into the high-frequency band.

    Algorithm:
        1. STFT the low-bitrate signal.
        2. Find the bin corresponding to cutoff_hz.
        3. Flip the spectrum below cutoff and paste above it (with gain).
        4. iSTFT back to waveform.
    """
    n_fft = 2048
    hop_length = 512

    D = np.fft.rfft(y_low, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    cutoff_bin = np.searchsorted(freqs, cutoff_hz)

    D_recon = D.copy()

    for k in range(1, fold_factor + 1):
        # mirror bins [0..cutoff] around cutoff, place them at [cutoff..2*cutoff], etc.
        low_slice = D[:cutoff_bin]
        folded = low_slice[::-1]  # mirror
        start = cutoff_bin * k
        end = start + len(folded)
        if end > len(D_recon):
            end = len(D_recon)
            folded = folded[: end - start]
        decay = hf_gain * (0.6 ** (k - 1))
        D_recon[start:end] += decay * folded

    # Frame-by-frame via overlap-add using STFT
    _, _, Zxx = signal.stft(y_low, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
    freqs_stft = np.linspace(0, sr / 2, Zxx.shape[0])
    cutoff_bin_stft = np.searchsorted(freqs_stft, cutoff_hz)

    Zxx_recon = Zxx.copy()
    for k in range(1, fold_factor + 1):
        low_slice = Zxx[:cutoff_bin_stft, :]
        folded = low_slice[::-1, :]
        start = cutoff_bin_stft * k
        end = start + folded.shape[0]
        if end > Zxx_recon.shape[0]:
            end = Zxx_recon.shape[0]
            folded = folded[: end - start, :]
        decay = hf_gain * (0.6 ** (k - 1))
        Zxx_recon[start:end, :] += decay * folded

    _, y_recon = signal.istft(Zxx_recon, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)

    # Trim / pad to original length
    y_recon = _match_length(y_recon, len(y_low))
    y_recon = _normalize(y_recon)
    return y_recon.astype(np.float32)


# ── 3. Harmonic Extension BWE ────────────────────────────────────────────────────
def reconstruct_bwe_harmonic(
    y_low: np.ndarray,
    sr: int,
    cutoff_hz: float,
    hf_gain: float = 0.5,
) -> np.ndarray:
    """
    Synthesise high-frequency harmonics from the fundamental.

    Algorithm:
        1. STFT the low-bitrate signal.
        2. Half-wave rectify to create odd+even harmonics (non-linear excitation).
        3. High-pass the rectified signal to extract only the new HF content.
        4. Mix back with the original low-bitrate signal.
    """
    n_fft = 2048
    hop_length = 512

    # Non-linear distortion → generates harmonics
    y_excitation = np.sign(y_low) * (np.abs(y_low) ** 0.5)

    # High-pass to keep only the newly generated HF energy
    nyq = sr / 2.0
    norm_cutoff = cutoff_hz / nyq
    norm_cutoff = np.clip(norm_cutoff, 0.001, 0.999)
    b, a = signal.butter(6, norm_cutoff, btype="high", analog=False)
    y_hf = signal.filtfilt(b, a, y_excitation)

    # Smooth energy envelope to avoid harsh transients
    window_size = int(sr * 0.02)  # 20 ms
    envelope_orig = _rms_envelope(y_low, window_size)
    envelope_hf = _rms_envelope(y_hf, window_size)

    # Scale HF so its energy matches the original low-freq envelope
    eps = 1e-8
    energy_ratio = (envelope_orig + eps) / (envelope_hf + eps)
    y_hf_scaled = y_hf * energy_ratio * hf_gain

    y_recon = y_low + y_hf_scaled
    y_recon = _match_length(y_recon, len(y_low))
    y_recon = _normalize(y_recon)
    return y_recon.astype(np.float32)


# ── helpers ──────────────────────────────────────────────────────────────────────
def _rms_envelope(y: np.ndarray, window: int) -> np.ndarray:
    """Compute frame-by-frame RMS envelope."""
    y_sq = y ** 2
    kernel = np.ones(window) / window
    rms = np.sqrt(np.convolve(y_sq, kernel, mode="same") + 1e-10)
    return rms


def _match_length(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) > target_len:
        return y[:target_len]
    elif len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)))
    return y


def _normalize(y: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    peak = np.max(np.abs(y))
    if peak > 1e-8:
        return y / peak * target_peak
    return y
