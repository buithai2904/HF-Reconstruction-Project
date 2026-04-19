import streamlit as st
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
import io
import os
import tempfile

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Audio HF Reconstruction",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
}

/* Header */
.main-header {
    background: linear-gradient(135deg, #0d0d0d 0%, #1a1a2e 50%, #16213e 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    border: 1px solid #00d4ff22;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, #00d4ff18 0%, transparent 70%);
    pointer-events: none;
}
.main-header h1 {
    color: #00d4ff !important;
    font-size: 1.8rem;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #8892a4;
    margin: 0;
    font-size: 0.9rem;
    font-weight: 300;
}

/* Metric cards */
.metric-card {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    color: #00d4ff;
    font-weight: 700;
}
.metric-label {
    font-size: 0.75rem;
    color: #8892a4;
    margin-top: 0.2rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Method badge */
.method-badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
}
.badge-bwe { background: #1f3a2f; color: #3fb950; border: 1px solid #3fb95044; }
.badge-dl  { background: #2d1f3a; color: #a78bfa; border: 1px solid #a78bfa44; }

/* Info box */
.info-box {
    background: #0d1117;
    border-left: 3px solid #00d4ff;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.85rem;
    color: #8892a4;
}
</style>
""", unsafe_allow_html=True)

# ── imports with error handling ─────────────────────────────────────────────────
DL_AVAILABLE = False
try:
    from dl_model import reconstruct_hf_dl
    DL_AVAILABLE = True
except Exception:
    pass

from bwe_methods import (
    apply_lowpass_filter,
    reconstruct_bwe_spectral_folding,
    reconstruct_bwe_harmonic,
)
from evaluation import compute_metrics
from visualization import (
    plot_waveform_comparison,
    plot_spectrogram_comparison,
    plot_frequency_response,
    plot_metrics_radar,
)


# ── helpers ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_audio(file_bytes, filename):
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[1], delete=False) as f:
        f.write(file_bytes)
        tmp_path = f.name
    y, sr = librosa.load(tmp_path, sr=None, mono=True)
    os.unlink(tmp_path)
    return y, sr


def audio_to_bytes(y, sr):
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    buf.seek(0)
    return buf.read()


# ── sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.divider()

    st.markdown("**Lowpass cutoff**")
    cutoff_freq = st.slider(
        "Simulate low-bitrate (Hz)", 1000, 8000, 4000, 500,
        help="Frequency cutoff applied to simulate low-bitrate audio"
    )

    st.markdown("**Reconstruction method**")
    method = st.radio(
        "Method", ["BWE — Spectral Folding", "BWE — Harmonic Extension", "Deep Learning (SEANet)"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("**BWE parameters**")
    fold_factor = st.slider("Fold factor", 1, 4, 2, help="How many times to mirror the spectrum")
    gain_db = st.slider("HF gain (dB)", -12, 6, -3)

    st.divider()
    st.markdown("""
    <div class='info-box'>
    <b>Methods</b><br>
    • <b>Spectral Folding</b>: mirrors the low-freq spectrum to fill HF bands<br>
    • <b>Harmonic Extension</b>: synthesises harmonics from the fundamental<br>
    • <b>SEANet</b>: deep learning encoder-decoder model
    </div>
    """, unsafe_allow_html=True)


# ── main header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🎵 Audio HF Reconstruction</h1>
  <p>Data Compression & Coding — Demo System &nbsp;|&nbsp; Compare low-bitrate vs reconstructed audio</p>
</div>
""", unsafe_allow_html=True)


# ── file upload ──────────────────────────────────────────────────────────────────
col_upload, col_info = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload audio file", type=["wav", "mp3", "flac", "ogg"],
        help="Recommended: speech audio, 16–44.1 kHz, mono or stereo"
    )

with col_info:
    st.markdown("""
    <div class='info-box' style='margin-top:1.8rem'>
    <b>Recommended input</b><br>
    Format: WAV / FLAC<br>
    Duration: 3–30 seconds<br>
    Content: speech or music
    </div>
    """, unsafe_allow_html=True)


# ── use demo audio if no upload ───────────────────────────────────────────────────
if uploaded_file is None:
    st.info("👆 Upload an audio file, or click below to use a generated demo signal.")
    if st.button("▶ Use demo signal (synthetic speech-like)"):
        sr_demo = 22050
        duration = 4.0
        t = np.linspace(0, duration, int(sr_demo * duration))
        # Simulate speech-like signal: sum of harmonics + noise
        freqs = [120, 240, 480, 960, 1920, 3840, 7680]
        amps  = [1.0, 0.7, 0.5, 0.35, 0.2, 0.1, 0.05]
        y_demo = sum(a * np.sin(2 * np.pi * f * t) for f, a in zip(freqs, amps))
        y_demo += 0.03 * np.random.randn(len(t))
        y_demo = (y_demo / np.max(np.abs(y_demo)) * 0.8).astype(np.float32)
        st.session_state["demo_audio"] = (y_demo, sr_demo)

    if "demo_audio" in st.session_state:
        y_orig, sr = st.session_state["demo_audio"]
    else:
        st.stop()
else:
    file_bytes = uploaded_file.read()
    with st.spinner("Loading audio..."):
        y_orig, sr = load_audio(file_bytes, uploaded_file.name)


# ── processing ────────────────────────────────────────────────────────────────────
with st.spinner("Processing audio..."):
    # 1. Simulate low-bitrate
    y_low = apply_lowpass_filter(y_orig, cutoff_freq, sr)

    # 2. Reconstruct
    gain_linear = 10 ** (gain_db / 20)

    if "Spectral Folding" in method:
        y_recon = reconstruct_bwe_spectral_folding(y_low, sr, cutoff_freq, fold_factor, gain_linear)
        badge = "<span class='method-badge badge-bwe'>BWE · Spectral Folding</span>"
    elif "Harmonic" in method:
        y_recon = reconstruct_bwe_harmonic(y_low, sr, cutoff_freq, gain_linear)
        badge = "<span class='method-badge badge-bwe'>BWE · Harmonic Extension</span>"
    else:
        if DL_AVAILABLE:
            y_recon = reconstruct_hf_dl(y_low, sr)
            badge = "<span class='method-badge badge-dl'>Deep Learning · SEANet</span>"
        else:
            st.warning("Deep Learning model not available — falling back to Spectral Folding.")
            y_recon = reconstruct_bwe_spectral_folding(y_low, sr, cutoff_freq, fold_factor, gain_linear)
            badge = "<span class='method-badge badge-bwe'>BWE · Spectral Folding (fallback)</span>"

    # 3. Compute metrics
    metrics = compute_metrics(y_orig, y_low, y_recon, sr)


# ── metrics row ───────────────────────────────────────────────────────────────────
st.markdown(f"**Active method:** {badge}", unsafe_allow_html=True)
st.markdown("")

m1, m2, m3, m4, m5 = st.columns(5)
cards = [
    (m1, f"{metrics['snr_low']:.1f} dB",   "SNR (degraded)"),
    (m2, f"{metrics['snr_recon']:.1f} dB",  "SNR (reconstructed)"),
    (m3, f"{metrics['lsd_low']:.2f}",       "LSD (degraded)"),
    (m4, f"{metrics['lsd_recon']:.2f}",     "LSD (reconstructed)"),
    (m5, f"{sr//1000} kHz",                 "Sample rate"),
]
for col, val, label in cards:
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{label}</div>
        </div>""", unsafe_allow_html=True)


# ── tabs ──────────────────────────────────────────────────────────────────────────
st.markdown("")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Spectrogram", "🌊 Waveform", "📈 Frequency Response", "🎧 Audio Player", "📂 Dataset Browser"
])

with tab1:
    st.markdown("#### Spectrogram comparison — low-bitrate vs reconstructed vs original")
    fig = plot_spectrogram_comparison(y_orig, y_low, y_recon, sr)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with tab2:
    st.markdown("#### Waveform comparison")
    fig = plot_waveform_comparison(y_orig, y_low, y_recon, sr)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with tab3:
    st.markdown("#### Frequency response — HF recovery visualised")
    fig = plot_frequency_response(y_orig, y_low, y_recon, sr, cutoff_freq)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with tab4:
    st.markdown("#### Listen and compare")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🔵 Original**")
        st.audio(audio_to_bytes(y_orig, sr), format="audio/wav")
    with c2:
        st.markdown("**🔴 Low-bitrate (degraded)**")
        st.audio(audio_to_bytes(y_low, sr), format="audio/wav")
    with c3:
        st.markdown("**🟢 Reconstructed**")
        st.audio(audio_to_bytes(y_recon, sr), format="audio/wav")

    st.divider()
    st.markdown("**Download reconstructed audio**")
    st.download_button(
        "⬇ Download WAV",
        data=audio_to_bytes(y_recon, sr),
        file_name="reconstructed_hf.wav",
        mime="audio/wav"
    )

with tab5:
    from dataset_browser import render_dataset_tab
    render_dataset_tab(data_dir="./data")