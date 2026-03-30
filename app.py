import gradio as gr
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch

# ── Load model at startup ─────────────────────────────────────
MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"

print(f"Loading model: {MODEL_NAME} …")
model     = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
device    = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded on {device} ✓")

# ── Inference ─────────────────────────────────────────────────
def generate_caption(image, max_length, num_beams, num_captions):
    if image is None:
        return "Please upload an image."

    image = image.convert("RGB")
    pixel_values = processor(images=[image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(
        pixel_values,
        max_length=int(max_length),
        num_beams=int(num_beams),
        num_return_sequences=int(num_captions),
        early_stopping=True,
    )

    captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    captions = [c.strip() for c in captions]

    if len(captions) == 1:
        return captions[0]
    return "\n\n".join([f"{i+1}. {c}" for i, c in enumerate(captions)])


# ── Gradio UI ─────────────────────────────────────────────────
with gr.Blocks(title="Image Captioning · ViT+GPT2") as demo:

    gr.Markdown("""
    # 🖼️ Image Captioning — ViT + GPT-2
    Upload any image and get an AI-generated caption.
    Model: [nlpconnect/vit-gpt2-image-captioning](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning)
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image", height=300)
            max_length  = gr.Slider(10, 128, value=64, step=1,
                                    label="Max caption length")
            num_beams   = gr.Slider(1, 8, value=4, step=1,
                                    label="Beam width (higher = more accurate)")
            num_captions = gr.Slider(1, 4, value=1, step=1,
                                     label="Number of captions")
            btn = gr.Button("→ Generate Caption", variant="primary")

        with gr.Column():
            output = gr.Textbox(label="Generated Caption", lines=5)
            gr.Markdown("""
            ### How it works
            - **ViT** encodes the image into patch embeddings
            - **GPT-2** decodes embeddings into natural language
            - **Beam search** picks the best caption from multiple candidates

            ### Tips
            - Clear, well-lit photos work best
            - Increase beam width for better accuracy
            - Multiple captions reveals model uncertainty
            """)

    btn.click(
        fn=generate_caption,
        inputs=[image_input, max_length, num_beams, num_captions],
        outputs=output,
    )

    gr.Markdown("---\nPart of the [AI Engineer Portfolio](https://github.com/amarshiv86)")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
