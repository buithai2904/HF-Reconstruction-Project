# 🎵 Audio HF Reconstruction — Demo System
**Data Compression & Coding Project**

Demo Streamlit so sánh audio low-bitrate vs. reconstructed bằng BWE truyền thống và Deep Learning.

---

## 🚀 Cách chạy (Local — khuyên dùng)

### Bước 1 — Cài Python 3.10+
Tải tại https://www.python.org/downloads/

### Bước 2 — Tạo virtual environment
```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Bước 3 — Cài dependencies
```bash
pip install -r requirements.txt
```

### Bước 4 — Chạy app
```bash
streamlit run app.py
```
→ Mở trình duyệt tại **http://localhost:8501**

---

## ☁️ Deploy lên Streamlit Cloud (free, không cần server)

1. Push toàn bộ folder này lên GitHub (public repo)
2. Vào https://share.streamlit.io → "New app"
3. Chọn repo → branch `main` → file `app.py`
4. Bấm **Deploy** → có link share ngay

---

## 🐳 Chạy bằng Docker (optional)

```bash
docker build -t audio-hf-demo .
docker run -p 8501:8501 audio-hf-demo
```

---

## 📁 Cấu trúc project

```
audio_bwe_demo/
├── app.py            ← Streamlit UI chính
├── bwe_methods.py    ← BWE: Spectral Folding + Harmonic Extension
├── dl_model.py       ← Deep Learning: EnCodec/SEANet (optional)
├── evaluation.py     ← Metrics: SNR, LSD
├── visualization.py  ← Matplotlib plots
├── requirements.txt
└── README.md
```

---

## 🧪 Enable Deep Learning (SEANet/EnCodec)

```bash
pip install torch torchaudio encodec
```
Sau đó chọn **"Deep Learning (SEANet)"** trong sidebar.

---

## 📊 Metrics được hiển thị

| Metric | Ý nghĩa | Tốt khi |
|--------|---------|---------|
| SNR    | Signal-to-Noise Ratio | Cao hơn |
| LSD    | Log-Spectral Distance | Thấp hơn |
