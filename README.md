---
title: Image Captioning ViT GPT2
emoji: 🖼️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.10.0
app_file: app.py
pinned: false
---

# 🖼️ Image Captioning — ViT + GPT-2

Upload any image and get an AI-generated caption using a
Vision Transformer encoder + GPT-2 decoder pipeline.

## Model
[`nlpconnect/vit-gpt2-image-captioning`](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning)

## Stack
| Layer | Technology |
|---|---|
| Vision encoder | ViT (Vision Transformer) |
| Text decoder | GPT-2 |
| UI | Gradio |
| Hosting | Hugging Face Spaces |

## Features
- Adjustable beam search width
- Temperature control for creativity
- Generate multiple captions at once
