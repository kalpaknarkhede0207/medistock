from pydantic import BaseModel, Field
from typing import List, Optional

class OcrDraftItemResponse(BaseModel):
    physicalRowId: str = Field(..., description="e.g. ROW_01")
    productName: str
    category: str = "Medicine"
    batchNumber: str = ""
    expiryDate: str = ""
    quantity: int = 1
    freeQuantity: int = 0
    purchaseRate: float = 0.0
    mrp: float = 0.0
    amount: float = 0.0
    discountPct: float = 0.0
    cgstRate: float = 0.0
    cgstAmt: float = 0.0
    sgstRate: float = 0.0
    sgstAmt: float = 0.0
    cropTopPercent: float = 0.0
    cropBottomPercent: float = 0.0

class OcrDraftBillResponse(BaseModel):
    supplierName: str = ""
    invoiceNo: str = ""
    billDate: str = ""
    totalAmount: float = 0.0
    parsedSource: str = "PaddleOCR + OpenCV Backend"
    items: List[OcrDraftItemResponse] = []
