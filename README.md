# 🖼️ Image Captioning — ViT + GPT-2

An image captioning pipeline using a **Vision Transformer (ViT)** encoder and **GPT-2**
decoder. Upload any image and get an AI-generated natural language description with
adjustable beam search and caption count controls.

---

## 🌐 Live Demo

👉 **[Try it live on HF Spaces](https://amarshiv86-image-captioning-vit-gpt2.hf.space)**

---

## 🏗️ How It Works

```
Image Input
    ↓
ViT — encodes image into patch embeddings
    ↓
GPT-2 — decodes embeddings into natural language tokens
    ↓
Beam Search — selects best caption from multiple candidates
    ↓
Natural Language Caption
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Vision encoder | ViT (Vision Transformer) |
| Text decoder | GPT-2 |
| Framework | HuggingFace Transformers 5.4.0 |
| UI | Gradio 6.10.0 |
| CI/CD | GitHub Actions |
| Hosting | Hugging Face Spaces |

---

## 📁 Project Structure

```
image-captioning/
├── app.py              # Gradio app — model loading + inference + UI
├── requirements.txt    # Pinned dependencies
├── .gitignore
└── .github/
    └── workflows/
        └── deploy.yml  # Auto-deploy to HF Spaces on push
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/amarshiv86/image-captioning
cd image-captioning
pip install -r requirements.txt
python app.py
# → open the gradio.live link printed in terminal
```

---

## ⚙️ Features

- **Beam search** width 1–8 — higher = more accurate captions
- **Multiple captions** — generate up to 4 to compare model confidence
- Auto-detects GPU (CUDA) or falls back to CPU
- No custom training — uses pre-trained model from HF Hub

---

## 🤖 Model

[`nlpconnect/vit-gpt2-image-captioning`](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning)
— a pre-trained VisionEncoderDecoderModel combining ViT and GPT-2,
fine-tuned on COCO image captions (~118k images).

---

## 🔗 Part of AI Engineer Portfolio

| Project | Description |
|---|---|
| P1 — [Weather Prediction](https://github.com/amarshiv86/weather-mlops-pipeline) | Classical ML + full MLOps pipeline |
| P2 — [Sentiment Analysis](https://github.com/amarshiv86/sentiment-analysis-mlops-pipeline) | Fine-tuned distilBERT + drift detection |
| P3 — Image Captioning (this repo) | Multi-modal ViT+GPT2 inference |

---

## 📄 License

MIT — free to use, modify, and distribute.
