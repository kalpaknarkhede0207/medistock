import os
# Disable MKLDNN/oneDNN to prevent PIR ConvertPirAttribute2RuntimeAttribute crash
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import io
import json
import time
import cv2
import gradio_client.utils as cu

# Resolve Gradio 4 OpenAPI schema bug where boolean additionalProperties crashed api_info
_orig_get_type = getattr(cu, "get_type", None)
if _orig_get_type:
    def _safe_get_type(schema):
        if isinstance(schema, bool):
            return "bool"
        return _orig_get_type(schema)
    cu.get_type = _safe_get_type

_orig_json_schema = getattr(cu, "_json_schema_to_python_type", None)
if _orig_json_schema:
    def _safe_json_schema_to_python_type(schema, defs):
        if isinstance(schema, bool):
            return "Any"
        return _orig_json_schema(schema, defs)
    cu._json_schema_to_python_type = _safe_json_schema_to_python_type

import spaces
import gradio as gr
from fastapi import File, UploadFile, HTTPException, Header, Response
from app.geometry import preprocess_and_correct_perspective
from app.ocr_engine import PaddleOcrManager
from app.table_reconstructor import TableReconstructor

# API Protection & Payload Boundaries
OCR_API_KEY = os.getenv("MEDISTOCK_API_KEY", "medistock_sec_9b7c841f3e2a05d8")
MAX_IMAGE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB

# Initialize PaddleOCR and Table Reconstructor
ocr_manager = PaddleOcrManager.get_instance()
table_reconstructor = TableReconstructor()

def health_check():
    return {
        "status": "ok",
        "ocr_engine": "PaddleOCR v5 (PaddlePaddle 3.3.1)",
        "preprocessing": "OpenCV Geometry & Perspective",
        "hosting": "Hugging Face Spaces (ZeroGPU Free)",
        "api_protection": "Active (X-Api-Key)",
        "timestamp": time.time()
    }

# Keep ZeroGPU happy with a lightweight AST check
@spaces.GPU
def _zero_gpu_init():
    return True

def run_ocr_pipeline(image_bytes: bytes):
    t0 = time.time()
    corrected_img, meta = preprocess_and_correct_perspective(image_bytes)
    t_geo = time.time()
    img_h, img_w = corrected_img.shape[:2]

    ocr_output = ocr_manager.predict(corrected_img)
    t_ocr = time.time()
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
    t_recon = time.time()
    timings = f"geo={round(t_geo - t0, 3)}s, ocr={round(t_ocr - t_geo, 3)}s ({len(texts)} texts), recon={round(t_recon - t_ocr, 3)}s, total={round(t_recon - t0, 3)}s"
    print(f"[Timing] {timings}")
    return draft_bill, timings

async def parse_bill_api(
    response: Response,
    file: UploadFile = File(...),
    x_api_key: str = Header(None, alias="X-Api-Key")
):
    # Verify API Key Protection
    if OCR_API_KEY and x_api_key != OCR_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Missing or invalid X-Api-Key header. Only authorized MediStock clients can access this OCR engine."
        )

    start_time = time.time()
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image uploaded.")

        if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Payload Too Large: Maximum allowed invoice image size is 15 MB (uploaded {len(image_bytes)} bytes)."
            )

        draft_bill, timings = run_ocr_pipeline(image_bytes)
        elapsed = round(time.time() - start_time, 2)
        response.headers["X-Process-Time"] = f"{elapsed}s"
        response.headers["X-OCR-Timings"] = timings
        print(f"[PaddleOCR] Successfully processed {file.filename} in {elapsed}s: {len(draft_bill.items)} items extracted.")
        return draft_bill.model_dump()

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        raise HTTPException(status_code=500, detail=f"PaddleOCR processing error: {str(e)} | Stack: {tb}")

def parse_invoice_gradio(image):
    if image is None:
        return "Please upload an invoice image."
    try:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", img_bgr)
        draft, _ = run_ocr_pipeline(buffer.tobytes())
        return json.dumps(draft.model_dump(), indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

# Build Gradio UI
demo = gr.Interface(
    fn=parse_invoice_gradio,
    inputs=gr.Image(type="numpy", label="Upload Pharmaceutical Invoice Image"),
    outputs=gr.Textbox(label="Extracted Draft Invoice JSON (24 Line Items)", lines=25),
    title="MediStock PaddleOCR Invoice Extraction Engine",
    description="High accuracy pharmaceutical invoice parser running with 16 GB RAM / ZeroGPU. Android REST API: POST /ocr/parse-bill"
)

# Attach REST routes to both initial App and runtime App.create_app
from gradio.routes import App

_orig_create_app = App.create_app
def _custom_create_app(blocks, *args, **kwargs):
    app = _orig_create_app(blocks, *args, **kwargs)
    app.add_api_route("/health", health_check, methods=["GET", "HEAD"])
    app.add_api_route("/ocr/parse-bill", parse_bill_api, methods=["POST"])
    return app
App.create_app = _custom_create_app

demo.app.add_api_route("/health", health_check, methods=["GET", "HEAD"])
demo.app.add_api_route("/ocr/parse-bill", parse_bill_api, methods=["POST"])

if __name__ == "__main__":
    demo.launch()
