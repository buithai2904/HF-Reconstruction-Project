"""
visualization.py
Matplotlib figures for the Streamlit demo.
All plots use a dark theme consistent with the app CSS.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
import librosa.display

# ── shared dark theme ────────────────────────────────────────────────────────────
DARK_BG   = "#0d1117"
GRID_COL  = "#21262d"
TEXT_COL  = "#8892a4"
ACCENT    = "#00d4ff"
COLOR_ORIG  = "#00d4ff"   # blue  — original
COLOR_LOW   = "#f85149"   # red   — degraded
COLOR_RECON = "#3fb950"   # green — reconstructed

def _apply_dark(fig, axes_list):
    fig.patch.set_facecolor(DARK_BG)
    for ax in axes_list:
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=TEXT_COL, labelsize=8)
        ax.xaxis.label.set_color(TEXT_COL)
        ax.yaxis.label.set_color(TEXT_COL)
        ax.title.set_color(TEXT_COL)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)
        ax.grid(True, color=GRID_COL, linewidth=0.5, linestyle="--", alpha=0.7)


# ── 1. Spectrogram comparison ────────────────────────────────────────────────────
def plot_spectrogram_comparison(
    y_orig, y_low, y_recon, sr,
    n_fft=2048, hop_length=512
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    fig.subplots_adjust(wspace=0.08, left=0.06, right=0.98, top=0.88, bottom=0.12)

    signals = [
        (y_orig,  f"Original ({sr//1000} kHz)",     "Blues"),
        (y_low,   "Low-bitrate (degraded)",          "Reds"),
        (y_recon, "Reconstructed (HF recovered)",    "Greens"),
    ]

    for ax, (y, title, cmap) in zip(axes, signals):
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)),
            ref=np.max
        )
        img = librosa.display.specshow(
            D, sr=sr, hop_length=hop_length,
            x_axis="time", y_axis="hz",
            ax=ax, cmap=cmap
        )
        ax.set_title(title, fontsize=9, pad=6, color=TEXT_COL)
        ax.set_ylim(0, sr // 2)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Frequency (Hz)" if ax is axes[0] else "", fontsize=8)

    _apply_dark(fig, axes)
    fig.suptitle("Spectrogram Comparison", color=ACCENT, fontsize=11, y=0.97,
                 fontfamily="monospace")
    return fig


# ── 2. Waveform comparison ───────────────────────────────────────────────────────
def plot_waveform_comparison(y_orig, y_low, y_recon, sr):
    max_samples = sr * 5  # show max 5 s
    y_orig  = y_orig[:max_samples]
    y_low   = y_low[:max_samples]
    y_recon = y_recon[:max_samples]
    t = np.arange(len(y_orig)) / sr

    fig, axes = plt.subplots(3, 1, figsize=(14, 5), sharex=True)
    fig.subplots_adjust(hspace=0.45, left=0.06, right=0.98, top=0.88, bottom=0.10)

    pairs = [
        (axes[0], y_orig,  COLOR_ORIG,  "Original"),
        (axes[1], y_low,   COLOR_LOW,   "Low-bitrate (degraded)"),
        (axes[2], y_recon, COLOR_RECON, "Reconstructed"),
    ]
    for ax, y, color, label in pairs:
        ax.plot(t, y, color=color, linewidth=0.6, alpha=0.85)
        ax.set_ylabel("Amplitude", fontsize=8)
        ax.set_title(label, fontsize=9, color=TEXT_COL, pad=3)
        ax.set_ylim(-1.05, 1.05)

    axes[-1].set_xlabel("Time (s)", fontsize=8)
    _apply_dark(fig, axes)
    fig.suptitle("Waveform Comparison", color=ACCENT, fontsize=11, y=0.97,
                 fontfamily="monospace")
    return fig


# ── 3. Frequency response ─────────────────────────────────────────────────────────
def plot_frequency_response(y_orig, y_low, y_recon, sr, cutoff_hz):
    n_fft = 4096
    window = np.hanning(min(n_fft, len(y_orig)))

    def mag_db(y):
        n = min(n_fft, len(y))
        w = np.hanning(n)
        Y = np.abs(np.fft.rfft(y[:n] * w, n=n_fft)) + 1e-10
        return 20 * np.log10(Y / np.max(Y))

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    fig, ax = plt.subplots(figsize=(13, 4))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.14)

    ax.plot(freqs, mag_db(y_orig),  color=COLOR_ORIG,  lw=1.2, label="Original",      alpha=0.9)
    ax.plot(freqs, mag_db(y_low),   color=COLOR_LOW,   lw=1.2, label="Low-bitrate",   alpha=0.9)
    ax.plot(freqs, mag_db(y_recon), color=COLOR_RECON, lw=1.2, label="Reconstructed", alpha=0.9, linestyle="--")

    # Cutoff marker
    ax.axvline(cutoff_hz, color="#f0e68c", lw=1.0, linestyle=":", alpha=0.8, label=f"Cutoff {cutoff_hz} Hz")

    # Shade recovered HF region
    ax.axvspan(cutoff_hz, sr / 2, alpha=0.06, color=COLOR_RECON, label="HF recovered region")

    ax.set_xlim(0, sr / 2)
    ax.set_ylim(-80, 5)
    ax.set_xlabel("Frequency (Hz)", fontsize=9)
    ax.set_ylabel("Magnitude (dB)",  fontsize=9)
    ax.set_title("Frequency Response — High-frequency Recovery", fontsize=10, color=TEXT_COL)

    legend = ax.legend(fontsize=8, framealpha=0.3, facecolor=DARK_BG,
                       edgecolor=GRID_COL, labelcolor=TEXT_COL)

    _apply_dark(fig, [ax])
    fig.suptitle("Frequency Response", color=ACCENT, fontsize=11, y=0.99,
                 fontfamily="monospace")
    return fig


# ── 4. Metrics radar (optional, unused in tabs but importable) ───────────────────
def plot_metrics_radar(metrics: dict):
    labels = ["SNR\nimprovement", "LSD\nimprovement", "HF\nrecovery"]
    values = [
        np.clip(metrics["snr_delta"] / 20, 0, 1),
        np.clip(metrics["lsd_delta"] / 10, 0, 1),
        0.65,  # placeholder — would need MUSHRA
    ]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={"polar": True})
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    ax.plot(angles, values, color=ACCENT, lw=2)
    ax.fill(angles, values, color=ACCENT, alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color=TEXT_COL, fontsize=8)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], color=TEXT_COL, fontsize=7)
    ax.spines["polar"].set_color(GRID_COL)
    ax.grid(color=GRID_COL, linewidth=0.5)
    ax.tick_params(colors=TEXT_COL)

    return fig
