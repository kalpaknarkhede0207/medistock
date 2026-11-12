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
        print("Initializing PaddleOCR Engine (Fast CPU inference, batched recognition)...")
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_limit_side_len=1280,
            text_recognition_batch_size=32,
            lang="en",
            enable_mkldnn=False
        )
        # Configure internal text recognition model for batch size 32
        try:
            p_pipe = getattr(self.ocr, "paddlex_pipeline", None)
            internal_pipe = getattr(p_pipe, "_pipeline", p_pipe)
            if internal_pipe and hasattr(internal_pipe, "text_rec_model"):
                rec_model = getattr(internal_pipe, "text_rec_model")
                if hasattr(rec_model, "batch_sampler"):
                    rec_model.batch_sampler.batch_size = 32
                    print("PaddleOCR: Configured text_rec_model.batch_sampler.batch_size = 32")
        except Exception as e:
            print(f"Notice: text_rec batch size adjustment: {e}")

        print("PaddleOCR Engine Ready!")

    def predict(self, image_np) -> Dict[str, Any]:
        """
        Runs OCR on numpy image and extracts rec_texts, rec_scores, rec_boxes, rec_polys.
        Explicitly disable doc preprocessor and textline orientation in predict()
        to bypass the 250+ sequential classification passes.
        """
        results = self.ocr.predict(
            image_np,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_limit_side_len=1280,
            text_det_thresh=0.35,
            text_det_box_thresh=0.65
        )
        if not results:
            return {"rec_texts": [], "rec_scores": [], "rec_boxes": [], "rec_polys": []}

        res = results[0]
        return {
            "rec_texts": res.get("rec_texts", []),
            "rec_scores": res.get("rec_scores", []),
            "rec_boxes": res.get("rec_boxes", []),
            "rec_polys": res.get("rec_polys", [])
        }
