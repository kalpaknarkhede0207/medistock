import io
import json
import time
import os
import cv2
import spaces
import gradio as gr
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.models import OcrDraftBillResponse
from app.geometry import preprocess_and_correct_perspective
from app.ocr_engine import PaddleOcrManager
from app.table_reconstructor import TableReconstructor

# Initialize PaddleOCR and Table Reconstructor
ocr_manager = PaddleOcrManager.get_instance()
table_reconstructor = TableReconstructor()

# 1. Create FastAPI Application
app = FastAPI(
    title="MediStock PaddleOCR Backend",
    description="High-accuracy pharmaceutical invoice extraction service",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "ocr_engine": "PaddleOCR v5 (PaddlePaddle 3.3.1)",
        "preprocessing": "OpenCV Geometry & Perspective",
        "hosting": "Hugging Face Spaces (16 GB RAM / ZeroGPU)",
        "timestamp": time.time()
    }

@spaces.GPU
def run_ocr_pipeline(image_bytes: bytes):
    corrected_img, meta = preprocess_and_correct_perspective(image_bytes)
    img_h, img_w = corrected_img.shape[:2]

    ocr_output = ocr_manager.predict(corrected_img)
    texts = ocr_output["rec_texts"]
    scores = ocr_output["rec_scores"]
    boxes = ocr_output["rec_boxes"]

    if not texts:
        raise HTTPException(status_code=422, detail="No readable text detected on this invoice image.")

    draft_bill = table_reconstructor.reconstruct_invoice(
        texts=texts,
        scores=scores,
        boxes=boxes,
        image_height=img_h,
        image_width=img_w
    )
    return draft_bill

@app.post("/ocr/parse-bill", response_model=OcrDraftBillResponse)
async def parse_bill_api(file: UploadFile = File(...)):
    start_time = time.time()
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image uploaded.")

        draft_bill = run_ocr_pipeline(image_bytes)
        elapsed = round(time.time() - start_time, 2)
        print(f"[PaddleOCR] Successfully processed {file.filename} in {elapsed}s: {len(draft_bill.items)} items extracted.")
        return draft_bill

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"PaddleOCR processing error: {str(e)}")

# 2. Gradio Web UI callback
@spaces.GPU
def parse_invoice_gradio(image):
    if image is None:
        return "Please upload an invoice image."
    try:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", img_bgr)
        draft = run_ocr_pipeline(buffer.tobytes())
        return json.dumps(draft.model_dump(), indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

# Build Gradio UI
demo = gr.Interface(
    fn=parse_invoice_gradio,
    inputs=gr.Image(type="numpy", label="Upload Pharmaceutical Invoice Image"),
    outputs=gr.Code(language="json", label="Extracted Draft Invoice JSON (24 Line Items)"),
    title="MediStock PaddleOCR Invoice Extraction Engine",
    description="High accuracy pharmaceutical invoice parser running with 16 GB RAM. Android REST API: POST /ocr/parse-bill"
)

# 3. Mount Gradio UI at root path
app = gr.mount_gradio_app(app, demo, path="/")
