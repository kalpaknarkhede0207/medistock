import io
import os
import time
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.models import OcrDraftBillResponse
from app.geometry import preprocess_and_correct_perspective
from app.ocr_engine import PaddleOcrManager
from app.table_reconstructor import TableReconstructor
from app.qwen_parser import QwenInvoiceParser

EXPECTED_API_KEY = os.environ.get("OCR_API_KEY", "medistock_sec_9b7c841f3e2a05d8")
MAX_PAYLOAD_SIZE = 15 * 1024 * 1024  # 15 MB payload ceiling to prevent OOM

async def verify_api_key(x_api_key: str = Header(None)):
    """Verifies that requests to the OCR backend originate from an authorized MediStock client."""
    if not x_api_key or x_api_key != EXPECTED_API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Forbidden: Invalid or missing X-Api-Key authentication header."
        )

app = FastAPI(
    title="MediStock PaddleOCR + OpenCV Engine",
    description="High-accuracy coordinate-aware pharmaceutical invoice extraction service",
    version="1.0.0"
)

# Enable CORS for Android clients and local network
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
ocr_manager = PaddleOcrManager.get_instance()
table_reconstructor = TableReconstructor()
qwen_parser = QwenInvoiceParser()

@app.get("/")
def root():
    return {
        "service": "MediStock PaddleOCR + Qwen Hybrid Engine",
        "status": "online",
        "endpoints": {
            "health": "GET /health",
            "parse_bill": "POST /ocr/parse-bill"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "ocr_engine": "PaddleOCR v5 (PaddlePaddle 3.3.1)",
        "preprocessing": "OpenCV Geometry & Perspective",
        "hybrid_qwen": "Enabled (Active)" if qwen_parser.is_configured() else "Disabled (PaddleOCR Native Mode)",
        "timestamp": time.time()
    }

@app.post("/ocr/parse-bill", response_model=OcrDraftBillResponse)
async def parse_bill(
    file: UploadFile = File(...),
    _auth: None = Depends(verify_api_key)
):
    start_time = time.time()
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image uploaded.")
        if len(image_bytes) > MAX_PAYLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Image size exceeds 15 MB limit ({len(image_bytes) / (1024 * 1024):.2f} MB)."
            )

        # 1. OpenCV Preprocessing & Perspective Transform
        corrected_img, meta = preprocess_and_correct_perspective(image_bytes)
        img_h, img_w = corrected_img.shape[:2]

        # 2. PaddleOCR Inference
        ocr_output = ocr_manager.predict(corrected_img)
        texts = ocr_output["rec_texts"]
        scores = ocr_output["rec_scores"]
        boxes = ocr_output["rec_boxes"]

        if not texts:
            raise HTTPException(status_code=422, detail="No readable text detected on this invoice image.")

        # 3. Hybrid Structuring Route (PaddleOCR + Qwen LLM)
        draft_bill = None
        if qwen_parser.is_configured():
            try:
                words = table_reconstructor.prepare_words(texts, scores, boxes)
                clustered_lines, header_summary, rows, sup, inv, bdate, tot = table_reconstructor.get_row_clusters_and_header(words)
                if clustered_lines:
                    llm_data = await qwen_parser.parse_ocr_text_with_qwen(clustered_lines, header_summary)
                    if llm_data:
                        draft_bill = table_reconstructor.reconcile_llm_result(
                            llm_data=llm_data,
                            rows=rows,
                            image_height=img_h,
                            fallback_supplier=sup,
                            fallback_invoice=inv,
                            fallback_date=bdate,
                            fallback_total=tot
                        )
                        if draft_bill:
                            print(f"[Hybrid] Successfully structured {len(draft_bill.items)} items via Qwen LLM.")
            except Exception as e:
                print(f"[Hybrid] Qwen structuring fallback: {e}")

        # 4. Fallback to Native Spatial Table Reconstructor
        if draft_bill is None:
            draft_bill = table_reconstructor.reconstruct_invoice(
                texts=texts,
                scores=scores,
                boxes=boxes,
                image_height=img_h,
                image_width=img_w
            )

        elapsed = round(time.time() - start_time, 2)
        print(f"[{draft_bill.parsedSource}] Processed {file.filename} in {elapsed}s: {len(draft_bill.items)} items.")
        return draft_bill

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"PaddleOCR processing error: {str(e)}")
