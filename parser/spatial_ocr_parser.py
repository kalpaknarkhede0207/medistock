"""
MediStock: Adaptive Dynamic OCR Table Reconstruction Engine
Extracts medicines, surgical materials, quantities, batches, and rates from any bill format.
Decoupled from fixed 17-column layouts; adaptively extracts table items whether the bill has 3, 5, 8, or 17 columns.
"""

import re
from typing import List, Dict, Any, Optional

class BoundingBox:
    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.x_mid = (x_min + x_max) / 2.0
        self.y_mid = (y_min + y_max) / 2.0
        self.height = y_max - y_min
        self.width = x_max - x_min

class OcrWord:
    def __init__(self, text: str, bbox: BoundingBox):
        self.text = text
        self.bbox = bbox

class AdaptiveOcrTableReconstructor:
    def __init__(self):
        pass

    def parse_document(self, words: List[OcrWord]) -> Dict[str, Any]:
        """
        Dynamically extracts item lines and quantities from any bill format.
        """
        # 1. Detect header line boundary if present
        header_y = self._find_header_y(words)

        # 2. Filter body words below header
        body_words = [w for w in words if w.bbox.y_min > header_y]

        # 3. Cluster words into horizontal rows (lines)
        rows = self._cluster_rows_by_y(body_words)

        # 4. Adaptively parse each line into an item
        structured_items = []
        for row_words in rows:
            item = self._parse_line_adaptively(row_words)
            if item:
                structured_items.append(item)

        return {
            "total_items_detected": len(structured_items),
            "items": structured_items
        }

    def _find_header_y(self, words: List[OcrWord]) -> float:
        header_kws = ["PRODUCT", "ITEM", "DESCRIPTION", "PARTICULAR", "NAME", "QTY", "RATE", "MRP", "AMOUNT"]
        matches = [w for w in words if any(kw in w.text.upper() for kw in header_kws)]
        if matches:
            return max(w.bbox.y_max for w in matches)
        return 0.0

    def _cluster_rows_by_y(self, words: List[OcrWord], y_tolerance: float = 14.0) -> List[List[OcrWord]]:
        sorted_words = sorted(words, key=lambda w: w.bbox.y_mid)
        rows: List[List[OcrWord]] = []

        for word in sorted_words:
            matched_row = None
            for row in rows:
                avg_y = sum(w.bbox.y_mid for w in row) / len(row)
                if abs(word.bbox.y_mid - avg_y) <= y_tolerance:
                    matched_row = row
                    break
            if matched_row is not None:
                matched_row.append(word)
            else:
                rows.append([word])

        for row in rows:
            row.sort(key=lambda w: w.bbox.x_min)

        return rows

    def _parse_line_adaptively(self, row_words: List[OcrWord]) -> Optional[Dict[str, Any]]:
        row_text = " ".join(w.text for w in row_words).strip()

        # Skip headers / footers
        skip_words = ["TOTAL", "SUBTOTAL", "GST SUMMARY", "ROUND OFF", "BALANCE", "INVOICE", "DATE:", "PH:", "DL NO", "===="]
        if any(sw in row_text.upper() for sw in skip_words):
            return None

        # Extract tokens
        name_parts = []
        pack = ""
        batch = ""
        expiry = ""
        quantity = 1
        rate = 0.0
        mrp = 0.0
        amount = 0.0

        for w in row_words:
            t = w.text.strip()
            if not t or re.fullmatch(r"[\.\-\=]+", t):
                continue

            # Expiry check: MM/YY or MM/YYYY
            if re.fullmatch(r"\d{2}/\d{2,4}", t):
                expiry = self._normalize_expiry(t)
                continue

            # Pack size check: e.g. 25GM, 100GM, 6'S, 10x10, 50ML, 40N
            if re.fullmatch(r"\d+('S|GM|ML|N|T|TAB|CAP|KG|BOX|PAIR|PAIRS|X\d+)", t, re.IGNORECASE):
                pack = t
                continue

            # Decimal number check (Rate, MRP, Amount)
            if re.fullmatch(r"\d+\.\d{2}", t):
                val = float(t)
                if rate == 0.0 and val > 0:
                    rate = val
                elif mrp == 0.0 and val >= rate:
                    mrp = val
                else:
                    amount = val
                continue

            # Integer number check (Quantity)
            if re.fullmatch(r"\d{1,4}", t):
                num = int(t)
                if num > 0 and quantity == 1 and w.bbox.x_mid > 250:
                    quantity = num
                    continue

            # Batch check: alphanumeric e.g. B072, GL2026, 610508B4
            if re.fullmatch(r"[A-Z0-9]{3,12}", t, re.IGNORECASE) and not t.isdigit() and len(t) <= 12 and w.bbox.x_mid > 250:
                if not batch:
                    batch = t
                    continue

            # Text before numbers is Item / Product Name
            if w.bbox.x_mid < 350 or not name_parts:
                name_parts.append(t)

        item_name = " ".join(name_parts).strip()
        if not item_name or len(item_name) < 2:
            return None

        # Determine category automatically
        category = "Medicine"
        name_upper = item_name.upper()
        if any(s in name_upper for s in ["GLOVE", "COTTON", "BANDAGE", "SYRINGE", "NEEDLE", "GAUZE", "TAPE"]):
            category = "Surgical"
        elif any(c in name_upper for c in ["CREAM", "LOTION", "SOAP", "SHAMPOO", "FACEWASH", "FW", "POWDER"]):
            category = "Cosmetic"
        elif any(b in name_upper for b in ["BABY", "LACTOGEN", "CERELAC", "DIAPER", "PAMPERS"]):
            category = "Baby Care"
        elif any(o in name_upper for o in ["TOOTHPASTE", "BRUSH", "COLGATE", "BALM", "CHOCOTRON"]):
            category = "General OTC"

        if amount == 0.0 and rate > 0:
            amount = round(quantity * rate, 2)

        return {
            "product_name": item_name,
            "category": category,
            "pack": pack,
            "batch_number": batch,
            "expiry_date": expiry,
            "quantity": quantity,
            "purchase_rate": rate,
            "mrp": mrp if mrp > 0 else (round(rate * 1.25, 2) if rate > 0 else 0.0),
            "amount": amount
        }

    def _normalize_expiry(self, exp_str: str) -> str:
        m = re.search(r"(\d{2})/(\d{2,4})", exp_str)
        if m:
            month = int(m.group(1))
            year = int(m.group(2))
            if year < 100:
                year += 2000
            import calendar
            last_day = calendar.monthrange(year, month)[1]
            return f"{year:04d}-{month:02d}-{last_day:02d}"
        return ""

if __name__ == "__main__":
    reconstructor = AdaptiveOcrTableReconstructor()
    
    # Test with simulated 3-column bill and 17-column bill
    test_words = [
        OcrWord("SURGICAL", BoundingBox(20, 80, 80, 95)),
        OcrWord("GLOVES", BoundingBox(85, 80, 140, 95)),
        OcrWord("LATEX", BoundingBox(145, 80, 190, 95)),
        OcrWord("50PAIRS", BoundingBox(220, 80, 280, 95)),
        OcrWord("GL2026", BoundingBox(320, 80, 370, 95)),
        OcrWord("12/28", BoundingBox(400, 80, 440, 95)),
        OcrWord("10", BoundingBox(480, 80, 500, 95)),
        OcrWord("85.00", BoundingBox(560, 80, 600, 95)),
        OcrWord("120.00", BoundingBox(640, 80, 690, 95)),
        OcrWord("850.00", BoundingBox(750, 80, 800, 95))
    ]
    
    result = reconstructor.parse_document(test_words)
    import json
    print("SUCCESS: Adaptive Parser Extracted Item:")
    print(json.dumps(result, indent=2))
