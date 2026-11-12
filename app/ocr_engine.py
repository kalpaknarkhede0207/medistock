import os

# Disable MKLDNN/oneDNN to prevent PIR ConvertPirAttribute2RuntimeAttribute crash
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddleocr import PaddleOCR
from typing import Dict, Any, List

class PaddleOcrManager:
    _instance = None

    @classmethod
    def get_instance(cls) -> "PaddleOcrManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        print("Initializing PaddleOCR Engine...")
        self.ocr = PaddleOCR(use_textline_orientation=True, lang="en", enable_mkldnn=False)
        print("PaddleOCR Engine Ready!")

    def predict(self, image_np) -> Dict[str, Any]:
        """
        Runs OCR on numpy image and extracts rec_texts, rec_scores, rec_boxes, rec_polys.
        """
        results = self.ocr.predict(image_np)
        if not results:
            return {"rec_texts": [], "rec_scores": [], "rec_boxes": [], "rec_polys": []}

        res = results[0]
        return {
            "rec_texts": res.get("rec_texts", []),
            "rec_scores": res.get("rec_scores", []),
            "rec_boxes": res.get("rec_boxes", []),
            "rec_polys": res.get("rec_polys", [])
        }
