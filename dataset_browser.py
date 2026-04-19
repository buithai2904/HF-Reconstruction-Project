"""
dataset_browser.py
==================
Streamlit page để browse và demo trực tiếp từ dataset VCTK đã xử lý.

Chạy độc lập:
    streamlit run dataset_browser.py

Hoặc import vào app.py chính:
    from dataset_browser import render_dataset_tab
    render_dataset_tab()
"""

import streamlit as st
import numpy as np
import io
import os
from pathlib import Path

import librosa
import soundfile as sf
import matplotlib.pyplot as plt

from prepare_dataset import load_manifest, load_pair


# ── Constants ──────────────────────────────────────────────────────────────────
DARK_BG    = "#0d1117"
GRID_COL   = "#21262d"
TEXT_COL   = "#8892a4"
COLOR_ORIG  = "#00d4ff"
COLOR_LOW   = "#f85149"


def audio_to_bytes(y: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    buf.seek(0)
    return buf.read()


def plot_spectrogram_pair(y_orig, y_low, sr):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3), sharey=True)
    fig.subplots_adjust(wspace=0.06, left=0.06, right=0.98, top=0.85, bottom=0.15)
    fig.patch.set_facecolor(DARK_BG)

    for ax, (y, title, cmap) in zip(axes, [
        (y_orig, "Original (full bandwidth)", "Blues"),
        (y_low,  "Low-bitrate (filtered)",    "Reds"),
    ]):
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(y, n_fft=1024, hop_length=256)), ref=np.max
        )
        librosa.display.specshow(D, sr=sr, hop_length=256,
                                 x_axis="time", y_axis="hz",
                                 ax=ax, cmap=cmap)
        ax.set_facecolor(DARK_BG)
        ax.set_title(title, fontsize=9, color=TEXT_COL, pad=4)
        ax.tick_params(colors=TEXT_COL, labelsize=8)
        ax.set_xlabel("Time (s)", fontsize=8, color=TEXT_COL)
        ax.set_ylabel("Frequency (Hz)" if ax is axes[0] else "", fontsize=8, color=TEXT_COL)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)

    return fig


# ── Main render function ───────────────────────────────────────────────────────
def render_dataset_tab(data_dir: str = "./data"):
    """
    Render the dataset browser tab.
    Call this from app.py inside a st.tab block.
    """
    st.markdown("#### VCTK Dataset Browser")
    st.markdown(
        "<div style='font-size:0.85rem;color:#8892a4'>"
        "Browse the preprocessed VCTK pairs. "
        "Run <code>python prepare_dataset.py</code> first to generate the dataset."
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Load manifest ──────────────────────────────────────────────────────────
    if not Path(data_dir, "manifest.json").exists():
        st.warning(
            "Dataset not found. Run the preprocessing script first:\n\n"
            "```bash\npython prepare_dataset.py --cutoff 4000 --max_files 200\n```"
        )
        return

    try:
        manifest = load_manifest(data_dir)
    except Exception as e:
        st.error(f"Failed to load manifest: {e}")
        return

    # ── Stats row ──────────────────────────────────────────────────────────────
    train_entries = [e for e in manifest if e["split"] == "train"]
    test_entries  = [e for e in manifest if e["split"] == "test"]
    speakers      = sorted(set(e["speaker"] for e in manifest))
    total_dur     = sum(e["duration_s"] for e in manifest)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, f"{len(manifest)}",           "Total clips"),
        (c2, f"{len(train_entries)} / {len(test_entries)}", "Train / Test"),
        (c3, f"{len(speakers)}",           "Speakers"),
        (c4, f"{total_dur/60:.1f} min",    "Total audio"),
    ]:
        col.markdown(
            f"<div style='background:#0d1117;border:1px solid #21262d;"
            f"border-radius:10px;padding:0.8rem 1rem;text-align:center'>"
            f"<div style='font-size:1.3rem;font-weight:700;color:#00d4ff'>{val}</div>"
            f"<div style='font-size:0.72rem;color:#8892a4;text-transform:uppercase;"
            f"letter-spacing:0.5px'>{label}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Filters ────────────────────────────────────────────────────────────────
    col_split, col_spk, col_dur = st.columns([1, 2, 2])

    with col_split:
        split_filter = st.selectbox("Split", ["all", "train", "test"])

    with col_spk:
        spk_filter = st.selectbox("Speaker", ["all"] + speakers)

    with col_dur:
        dur_range = st.slider("Duration (s)", 0.0, 30.0, (0.0, 30.0), 0.5)

    # Apply filters
    filtered = [
        e for e in manifest
        if (split_filter == "all" or e["split"] == split_filter)
        and (spk_filter == "all" or e["speaker"] == spk_filter)
        and (dur_range[0] <= e["duration_s"] <= dur_range[1])
    ]

    st.markdown(f"**{len(filtered)} clips** match filters")

    if not filtered:
        st.info("No clips match the current filters.")
        return

    # ── Clip selector ──────────────────────────────────────────────────────────
    st.divider()
    clip_labels = [
        f"[{e['split']}] {e['stem']} — {e['speaker']} — {e['duration_s']:.1f}s"
        for e in filtered[:100]   # cap at 100 in dropdown
    ]
    selected_idx = st.selectbox("Select clip to preview", range(len(clip_labels)),
                                format_func=lambda i: clip_labels[i])
    entry = filtered[selected_idx]

    # ── Load and display ───────────────────────────────────────────────────────
    with st.spinner("Loading audio pair..."):
        try:
            y_orig, y_low, sr = load_pair(entry, data_dir)
        except Exception as e:
            st.error(f"Could not load audio: {e}")
            return

    # Metadata
    st.markdown(
        f"<div style='background:#0d1117;border-left:3px solid #00d4ff;"
        f"padding:0.6rem 1rem;border-radius:0 8px 8px 0;font-size:0.82rem;color:#8892a4'>"
        f"Speaker: <b style='color:#e6edf3'>{entry['speaker']}</b> &nbsp;|&nbsp; "
        f"Duration: <b style='color:#e6edf3'>{entry['duration_s']:.2f}s</b> &nbsp;|&nbsp; "
        f"Sample rate: <b style='color:#e6edf3'>{sr} Hz</b> &nbsp;|&nbsp; "
        f"Samples: <b style='color:#e6edf3'>{len(y_orig):,}</b>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Spectrogram
    fig = plot_spectrogram_pair(y_orig, y_low, sr)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Audio players
    c_orig, c_low = st.columns(2)
    with c_orig:
        st.markdown("**Original (full bandwidth)**")
        st.audio(audio_to_bytes(y_orig, sr), format="audio/wav")
    with c_low:
        st.markdown("**Low-bitrate (filtered)**")
        st.audio(audio_to_bytes(y_low, sr), format="audio/wav")

    # Download
    st.download_button(
        "⬇ Download low-bitrate WAV",
        data=audio_to_bytes(y_low, sr),
        file_name=f"{entry['stem']}_low.wav",
        mime="audio/wav",
    )

    # ── Dataset stats table ────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Speaker distribution")
    spk_counts = {}
    for e in manifest:
        spk_counts[e["speaker"]] = spk_counts.get(e["speaker"], 0) + 1

    top_spk = sorted(spk_counts.items(), key=lambda x: -x[1])[:15]
    spk_names = [s for s, _ in top_spk]
    spk_vals  = [c for _, c in top_spk]

    fig2, ax = plt.subplots(figsize=(12, 2.5))
    fig2.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    bars = ax.bar(spk_names, spk_vals, color="#378ADD", width=0.6)
    ax.set_xlabel("Speaker ID", fontsize=9, color=TEXT_COL)
    ax.set_ylabel("Clips",      fontsize=9, color=TEXT_COL)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(axis="y", color=GRID_COL, linewidth=0.5, linestyle="--", alpha=0.6)
    if len(speakers) > 15:
        ax.set_title(f"Top 15 of {len(speakers)} speakers", fontsize=9, color=TEXT_COL, pad=4)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)


# ── Standalone run ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    st.set_page_config(page_title="VCTK Dataset Browser", layout="wide")
    st.title("VCTK Dataset Browser")
    render_dataset_tab()