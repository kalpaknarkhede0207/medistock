import io
import json
import os
import cv2
import numpy as np
import gradio as gr
from fastapi import FastAPI
from app.main import app as fastapi_app
from app.geometry import preprocess_and_correct_perspective
from app.ocr_engine import PaddleOcrManager
from app.table_reconstructor import TableReconstructor

# Function for Gradio interactive Web UI
def parse_invoice_gradio(image):
    if image is None:
        return "Please upload an invoice photo."
    try:
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", img_bgr)
        corrected_img, meta = preprocess_and_correct_perspective(buffer.tobytes())
        img_h, img_w = corrected_img.shape[:2]

        ocr_output = PaddleOcrManager.get_instance().predict(corrected_img)
        table_recon = TableReconstructor()
        draft = table_recon.reconstruct_invoice(
            texts=ocr_output["rec_texts"],
            scores=ocr_output["rec_scores"],
            boxes=ocr_output["rec_boxes"],
            image_height=img_h,
            image_width=img_w
        )
        return json.dumps(draft.model_dump(), indent=2)
    except Exception as e:
        return f"Error processing invoice: {str(e)}"

# Create Gradio UI
demo = gr.Interface(
    fn=parse_invoice_gradio,
    inputs=gr.Image(type="numpy", label="Upload Pharmaceutical Invoice"),
    outputs=gr.Code(language="json", label="Extracted Line Items (24 Items Accurate)"),
    title="MediStock PaddleOCR Pharmaceutical Invoice Parser",
    description="Production-grade invoice extraction running with 16 GB RAM on Hugging Face Spaces. Supports Android REST API at /ocr/parse-bill."
)

# Mount Gradio UI directly onto the FastAPI application
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
