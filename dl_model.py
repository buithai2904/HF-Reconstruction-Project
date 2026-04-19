"""
dl_model.py
Deep Learning reconstruction — SEANet / EnCodec based.

HOW TO ACTIVATE:
    pip install encodec torch torchaudio

Always uses the 24kHz mono model — the 48kHz model requires stereo (2 channels)
and is incompatible with mono audio input.
"""

import numpy as np

try:
    import torch
    import torchaudio
    from encodec import EncodecModel

    _model_cache = {}

    def _get_model():
        # Always use 24kHz model: mono (1 channel), no stereo requirement
        if "24k" not in _model_cache:
            model = EncodecModel.encodec_model_24khz()
            model.set_target_bandwidth(6.0)  # 6 kbps
            model.eval()
            _model_cache["24k"] = model
        return _model_cache["24k"]

    def reconstruct_hf_dl(y_low: np.ndarray, sr: int) -> np.ndarray:
        """
        Encode with EnCodec 24kHz (mono) at 6 kbps, then decode.

        Root cause of channel error: the 48kHz model expects 2 channels (stereo).
        Fix: always use 24kHz model which accepts 1 channel (mono).

        encodec >= 0.1.1: encode() returns list of EncodedFrame namedtuples.
        Pass that same list directly to decode().
        """
        TARGET_SR = 24000

        model = _get_model()

        # Step 1: build tensor [1, 1, T] and resample to 24kHz
        wav = torch.tensor(y_low, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)

        # Step 2: encode → decode
        with torch.no_grad():
            encoded_frames = model.encode(wav)
            decoded = model.decode(encoded_frames)  # [1, 1, T']

        # Step 3: to numpy, remove batch+channel dims → [T']
        y_out = decoded.squeeze().cpu().numpy()

        # Step 4: resample back to original sr
        if sr != TARGET_SR:
            wav_out = torch.tensor(y_out).unsqueeze(0).unsqueeze(0)
            wav_out = torchaudio.functional.resample(wav_out, TARGET_SR, sr)
            y_out = wav_out.squeeze().numpy()

        # Step 5: trim / pad to original length
        n = len(y_low)
        if len(y_out) > n:
            y_out = y_out[:n]
        elif len(y_out) < n:
            y_out = np.pad(y_out, (0, n - len(y_out)))

        # Step 6: normalise peak to 0.9
        peak = np.max(np.abs(y_out))
        if peak > 1e-8:
            y_out = y_out / peak * 0.9

        return y_out.astype(np.float32)

except ImportError:
    def reconstruct_hf_dl(y_low: np.ndarray, sr: int) -> np.ndarray:
        raise RuntimeError("torch and encodec are required for DL reconstruction.")