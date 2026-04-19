"""
prepare_dataset.py
==================
VCTK Corpus preprocessing pipeline for the Audio HF Reconstruction project.

Steps:
  1. Download VCTK from Kaggle via kagglehub
  2. Scan and audit the dataset structure
  3. Apply low-pass filter to simulate low-bitrate audio
  4. Save (original, low-bitrate) pairs into a clean output folder
  5. Print a summary report

Usage:
  python prepare_dataset.py [--cutoff 4000] [--max_files 200] [--output ./data]

Requirements:
  pip install kagglehub librosa soundfile scipy tqdm numpy
"""

import argparse
import os
import json
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import signal
import soundfile as sf
import librosa
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ── CLI args ──────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="VCTK preprocessing for BWE project")
    p.add_argument("--cutoff",    type=int,   default=4000,
                   help="Low-pass cutoff frequency in Hz (default: 4000)")
    p.add_argument("--order",     type=int,   default=8,
                   help="Butterworth filter order (default: 8, matches bwe_methods.py)")
    p.add_argument("--max_files", type=int,   default=200,
                   help="Max audio files to process (default: 200, use 0 for all)")
    p.add_argument("--target_sr", type=int,   default=22050,
                   help="Resample all audio to this rate (default: 22050)")
    p.add_argument("--min_dur",   type=float, default=1.0,
                   help="Minimum clip duration in seconds to keep (default: 1.0)")
    p.add_argument("--max_dur",   type=float, default=30.0,
                   help="Maximum clip duration in seconds to keep (default: 30.0)")
    p.add_argument("--output",    type=str,   default="./data",
                   help="Output directory (default: ./data)")
    p.add_argument("--split",     type=float, default=0.8,
                   help="Train fraction for train/test split (default: 0.8)")
    return p.parse_args()


# ── Step 1: Download dataset ──────────────────────────────────────────────────
def download_vctk():
    print("\n[1/5] Downloading VCTK Corpus from Kaggle...")
    try:
        import kagglehub
        path = kagglehub.dataset_download("pratt3000/vctk-corpus")
        print(f"      Downloaded to: {path}")
        return Path(path)
    except Exception as e:
        print(f"      ERROR: {e}")
        print("      Make sure kagglehub is installed and Kaggle API credentials are set.")
        print("      See: https://github.com/Kaggle/kagglehub#api-credentials")
        raise


# ── Step 2: Scan dataset structure ───────────────────────────────────────────
def scan_dataset(vctk_root: Path):
    print("\n[2/5] Scanning dataset structure...")

    # VCTK on Kaggle typically has wav48/ or wav16/ subfolder
    # Try common layouts
    audio_dirs = []
    for candidate in ["wav48", "wav48_silence_trimmed", "wav16", "wav", "."]:
        d = vctk_root / candidate
        if d.exists():
            audio_dirs.append(d)

    all_wav = []
    for d in audio_dirs:
        found = list(d.rglob("*.wav")) + list(d.rglob("*.flac"))
        if found:
            all_wav.extend(found)
            print(f"      Found {len(found):>5} files in: {d.relative_to(vctk_root)}")

    if not all_wav:
        # Flat search from root
        all_wav = list(vctk_root.rglob("*.wav")) + list(vctk_root.rglob("*.flac"))
        print(f"      Flat search found {len(all_wav)} files")

    # Deduplicate
    all_wav = sorted(set(all_wav))

    # Speaker breakdown
    speakers = set()
    for f in all_wav:
        # VCTK naming: p225_001.wav → speaker p225
        parts = f.stem.split("_")
        if parts:
            speakers.add(parts[0])

    print(f"\n      Total audio files : {len(all_wav)}")
    print(f"      Unique speakers   : {len(speakers)}")
    print(f"      Example files     :")
    for f in all_wav[:3]:
        print(f"        {f.name}")

    return all_wav, sorted(speakers)


# ── Step 3: Low-pass filter (exact same as bwe_methods.py) ───────────────────
def apply_lowpass_filter(y: np.ndarray, cutoff_hz: float, sr: int, order: int = 8) -> np.ndarray:
    """
    Identical implementation to bwe_methods.apply_lowpass_filter().
    Butterworth LP, zero-phase (filtfilt), order=8 by default.
    """
    nyq = sr / 2.0
    norm_cutoff = np.clip(cutoff_hz / nyq, 0.001, 0.999)
    b, a = signal.butter(order, norm_cutoff, btype="low", analog=False)
    y_filtered = signal.filtfilt(b, a, y)
    return y_filtered.astype(np.float32)


# ── Step 4: Process files ─────────────────────────────────────────────────────
def process_files(
    all_wav,
    output_dir: Path,
    cutoff_hz: int,
    order: int,
    target_sr: int,
    min_dur: float,
    max_dur: float,
    max_files: int,
    split: float,
):
    print(f"\n[3/5] Processing files...")
    print(f"      Cutoff    : {cutoff_hz} Hz")
    print(f"      Filter    : Butterworth order {order}, zero-phase (filtfilt)")
    print(f"      Target SR : {target_sr} Hz")
    print(f"      Duration  : {min_dur}s – {max_dur}s")

    # Output folder structure
    for split_name in ["train", "test"]:
        (output_dir / split_name / "original").mkdir(parents=True, exist_ok=True)
        (output_dir / split_name / "low_bitrate").mkdir(parents=True, exist_ok=True)

    # Shuffle for reproducibility then limit
    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(len(all_wav))
    if max_files > 0:
        indices = indices[:max_files]

    # Train / test split
    n_train = int(len(indices) * split)
    train_indices = set(indices[:n_train].tolist())
    test_indices  = set(indices[n_train:].tolist())

    stats = {
        "processed": 0,
        "skipped_duration": 0,
        "skipped_error": 0,
        "train": 0,
        "test": 0,
        "speakers": set(),
        "total_seconds": 0.0,
    }

    manifest = []  # [{split, stem, speaker, duration_s, sr}]

    all_indices = sorted(set(train_indices) | set(test_indices))

    for idx in tqdm(all_indices, desc="Processing", unit="file"):
        fpath = all_wav[idx]
        split_name = "train" if idx in train_indices else "test"

        try:
            # Load audio (mono, native sr first for duration check)
            y_native, sr_native = librosa.load(fpath, sr=None, mono=True)
            duration = len(y_native) / sr_native

            # Duration filter
            if duration < min_dur or duration > max_dur:
                stats["skipped_duration"] += 1
                continue

            # Resample to target_sr if needed
            if sr_native != target_sr:
                y = librosa.resample(y_native, orig_sr=sr_native, target_sr=target_sr)
            else:
                y = y_native.astype(np.float32)

            # Normalise to peak 0.9
            peak = np.max(np.abs(y))
            if peak > 1e-8:
                y = y / peak * 0.9

            # Apply low-pass filter
            y_low = apply_lowpass_filter(y, cutoff_hz, target_sr, order)

            # Save pair
            stem = fpath.stem
            orig_path = output_dir / split_name / "original"    / f"{stem}.wav"
            low_path  = output_dir / split_name / "low_bitrate" / f"{stem}_low{cutoff_hz}hz.wav"

            sf.write(str(orig_path), y,     target_sr, subtype="PCM_16")
            sf.write(str(low_path),  y_low, target_sr, subtype="PCM_16")

            # Track speaker
            speaker = stem.split("_")[0] if "_" in stem else "unknown"
            stats["speakers"].add(speaker)
            stats["total_seconds"] += duration
            stats["processed"] += 1
            stats[split_name] += 1

            manifest.append({
                "split":      split_name,
                "stem":       stem,
                "speaker":    speaker,
                "duration_s": round(duration, 3),
                "sr":         target_sr,
                "original":   str(orig_path.relative_to(output_dir)),
                "low_bitrate":str(low_path.relative_to(output_dir)),
            })

        except Exception as e:
            stats["skipped_error"] += 1
            tqdm.write(f"  SKIP {fpath.name}: {e}")

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return stats, manifest, manifest_path


# ── Step 5: Summary ───────────────────────────────────────────────────────────
def print_summary(stats, manifest, output_dir, cutoff_hz, target_sr, manifest_path):
    print(f"\n[4/5] Dataset summary")
    print(f"      ─────────────────────────────────────────")
    print(f"      Files processed   : {stats['processed']}")
    print(f"      ├─ train          : {stats['train']}")
    print(f"      └─ test           : {stats['test']}")
    print(f"      Skipped (duration): {stats['skipped_duration']}")
    print(f"      Skipped (error)   : {stats['skipped_error']}")
    print(f"      Unique speakers   : {len(stats['speakers'])}")
    print(f"      Total audio       : {stats['total_seconds'] / 60:.1f} min")
    print(f"      Sample rate       : {target_sr} Hz")
    print(f"      LP cutoff         : {cutoff_hz} Hz")
    print(f"      Manifest          : {manifest_path}")
    print(f"      ─────────────────────────────────────────")

    print(f"\n[5/5] Output folder structure:")
    print(f"      {output_dir}/")
    print(f"      ├── manifest.json          ← index of all pairs")
    print(f"      ├── train/")
    print(f"      │   ├── original/          ← full-bandwidth WAV")
    print(f"      │   └── low_bitrate/       ← low-pass filtered WAV")
    print(f"      └── test/")
    print(f"          ├── original/")
    print(f"          └── low_bitrate/")

    # Sample entries
    if manifest:
        print(f"\n      Sample manifest entries:")
        for entry in manifest[:3]:
            print(f"        [{entry['split']}] {entry['stem']} "
                  f"({entry['speaker']}, {entry['duration_s']:.1f}s)")


# ── Streamlit loader (importable by app.py) ───────────────────────────────────
def load_manifest(data_dir: str = "./data"):
    """
    Load the manifest.json produced by this script.
    Returns a list of dicts, each with keys:
        split, stem, speaker, duration_s, sr, original, low_bitrate
    """
    manifest_path = Path(data_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. "
            "Run prepare_dataset.py first."
        )
    with open(manifest_path) as f:
        return json.load(f)


def load_pair(entry: dict, data_dir: str = "./data"):
    """
    Given a manifest entry, return (y_orig, y_low, sr) as numpy arrays.
    """
    base = Path(data_dir)
    y_orig, sr = librosa.load(str(base / entry["original"]),    sr=None, mono=True)
    y_low,  _  = librosa.load(str(base / entry["low_bitrate"]), sr=None, mono=True)
    return y_orig.astype(np.float32), y_low.astype(np.float32), sr


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # 1. Download
    vctk_root = download_vctk()

    # 2. Scan
    all_wav, speakers = scan_dataset(vctk_root)
    if not all_wav:
        print("ERROR: No audio files found. Check the Kaggle dataset structure.")
        exit(1)

    # 3-4. Process
    stats, manifest, manifest_path = process_files(
        all_wav      = all_wav,
        output_dir   = output_dir,
        cutoff_hz    = args.cutoff,
        order        = args.order,
        target_sr    = args.target_sr,
        min_dur      = args.min_dur,
        max_dur      = args.max_dur,
        max_files    = args.max_files,
        split        = args.split,
    )

    # 5. Summary
    print_summary(stats, manifest, output_dir, args.cutoff, args.target_sr, manifest_path)

    elapsed = time.time() - t0
    print(f"\n      Done in {elapsed:.1f}s\n")