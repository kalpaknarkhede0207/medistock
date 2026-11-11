import re
from typing import List, Dict, Any, Tuple, Optional
from app.models import OcrDraftBillResponse, OcrDraftItemResponse

class TableReconstructor:
    def __init__(self):
        pass

    def reconstruct_invoice(
        self,
        texts: List[str],
        scores: List[float],
        boxes: List[List[float]],
        image_height: int,
        image_width: int
    ) -> OcrDraftBillResponse:
        """
        Reconstructs pharmaceutical invoice from OCR text segments and coordinates.
        """
        # Pack items into structured word objects
        words = []
        for text, score, box in zip(texts, scores, boxes):
            x_min, y_min, x_max, y_max = box
            words.append({
                "text": text.strip(),
                "score": score,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "x_mid": (x_min + x_max) / 2.0,
                "y_mid": (y_min + y_max) / 2.0,
                "height": y_max - y_min,
                "width": x_max - x_min
            })

        # 1. Extract Invoice Header Fields (Supplier, Invoice No, Date)
        supplier_name = self._extract_supplier_name(words)
        invoice_no = self._extract_invoice_number(words)
        bill_date = self._extract_bill_date(words)

        # 2. Locate Table Header Y-coordinate
        table_header_y = self._find_table_header_y(words)

        # 3. Locate Footer / Total Y-coordinate
        footer_y, total_amount = self._find_footer_y_and_total(words, table_header_y)

        # 4. Filter words strictly within the table body (allow bottom item line)
        body_words = [
            w for w in words
            if w["y_min"] >= table_header_y and (footer_y == 0 or w["y_min"] < footer_y + 15)
        ]

        # 5. Cluster words into physical rows by Y overlap
        rows = self._cluster_rows_by_y(body_words, y_tolerance=9.5)

        # 6. Parse physical rows and stitch multi-line product names
        parsed_items: List[OcrDraftItemResponse] = []
        current_item: Optional[Dict[str, Any]] = None

        row_index = 1
        for row_words in rows:
            line_data = self._parse_single_row(row_words, image_height)
            if not line_data:
                continue

            # Check if this row is a continuation (wrapped description e.g. "NASAL SOLN", "DROPS")
            if line_data.get("is_continuation") and current_item is not None:
                current_item["productName"] += " " + line_data["continuation_text"]
                # Update bottom crop boundary to include wrapped line
                current_item["cropBottomPercent"] = max(current_item["cropBottomPercent"], line_data["crop_bottom"])
            else:
                if current_item is not None:
                    parsed_items.append(OcrDraftItemResponse(
                        physicalRowId=f"ROW_{row_index:02d}",
                        productName=current_item["productName"],
                        category=current_item["category"],
                        batchNumber=current_item["batchNumber"],
                        expiryDate=current_item["expiryDate"],
                        quantity=current_item["quantity"],
                        freeQuantity=current_item["freeQuantity"],
                        purchaseRate=current_item["purchaseRate"],
                        mrp=current_item["mrp"],
                        amount=current_item["amount"],
                        cropTopPercent=current_item["cropTopPercent"],
                        cropBottomPercent=current_item["cropBottomPercent"]
                    ))
                    row_index += 1

                current_item = {
                    "productName": line_data["product_name"],
                    "category": line_data["category"],
                    "batchNumber": line_data["batch_number"],
                    "expiryDate": line_data["expiry_date"],
                    "quantity": line_data["quantity"],
                    "freeQuantity": line_data["free_quantity"],
                    "purchaseRate": line_data["purchase_rate"],
                    "mrp": line_data["mrp"],
                    "amount": line_data["amount"],
                    "cropTopPercent": line_data["crop_top"],
                    "cropBottomPercent": line_data["crop_bottom"]
                }

        # Append last pending item
        if current_item is not None:
            parsed_items.append(OcrDraftItemResponse(
                physicalRowId=f"ROW_{row_index:02d}",
                productName=current_item["productName"],
                category=current_item["category"],
                batchNumber=current_item["batchNumber"],
                expiryDate=current_item["expiryDate"],
                quantity=current_item["quantity"],
                freeQuantity=current_item["freeQuantity"],
                purchaseRate=current_item["purchaseRate"],
                mrp=current_item["mrp"],
                amount=current_item["amount"],
                cropTopPercent=current_item["cropTopPercent"],
                cropBottomPercent=current_item["cropBottomPercent"]
            ))

        # If total_amount wasn't explicitly found in footer, compute from items
        if total_amount == 0.0 and parsed_items:
            total_amount = round(sum(it.amount for it in parsed_items), 2)

        return OcrDraftBillResponse(
            supplierName=supplier_name,
            invoiceNo=invoice_no,
            billDate=bill_date,
            totalAmount=total_amount,
            parsedSource="PaddleOCR v5 + OpenCV Server",
            items=parsed_items
        )

    def _extract_supplier_name(self, words: List[Dict[str, Any]]) -> str:
        for w in words[:15]:
            t = w["text"].upper()
            if any(k in t for k in ["M/S", "PHARMA", "DISTRIBUTOR", "AGENCIES", "ENTERPRISES", "MEDICAL", "HEALTHCARE"]):
                # Clean up prefix like "M/S"
                return w["text"].strip()
        if words:
            return words[0]["text"].strip()
        return "Pharmaceutical Supplier"

    def _extract_invoice_number(self, words: List[Dict[str, Any]]) -> str:
        for i, w in enumerate(words[:40]):
            t = w["text"]
            # Standalone invoice code pattern like F/383149 or INV-2024
            if re.fullmatch(r"[A-Z]{1,4}[/-]\d{4,10}", t):
                return t
            # Check for label "Bill No" or "Invoice No" and take adjacent token
            if re.search(r"\b(?:Bill|Invoice|Inv)\s*(?:No|#)?\b", t, re.IGNORECASE):
                parts = re.split(r"[:\s]+", t)
                if len(parts) > 1 and parts[-1].upper() not in ("NO", "BILL", "INV", "INVOICE", ""):
                    return parts[-1]
                if i + 1 < len(words):
                    next_t = words[i + 1]["text"].strip()
                    if next_t.upper() not in ("NO", "DATE", "TIME") and len(next_t) >= 2:
                        return next_t
        return ""

    def _extract_bill_date(self, words: List[Dict[str, Any]]) -> str:
        for i, w in enumerate(words[:40]):
            t = w["text"]
            date_match = re.search(r"\b(\d{2}[-/.]\d{2}[-/.]\d{2,4})\b", t)
            if date_match:
                return date_match.group(1)
        return ""

    def _find_table_header_y(self, words: List[Dict[str, Any]]) -> float:
        header_kws = ["PRODUCT", "ITEM", "PARTICULAR", "BATCH", "EXP", "QTY", "RATE", "MRP", "AMOUNT", "HSN"]
        for w in words:
            t = w["text"].upper()
            matched = sum(1 for kw in header_kws if kw in t)
            if matched >= 1 and w["y_min"] > 150:
                return w["y_max"]
        return 200.0

    def _find_footer_y_and_total(self, words: List[Dict[str, Any]], header_y: float) -> Tuple[float, float]:
        footer_y = 0.0
        total_amount = 0.0

        for i, w in enumerate(words):
            if w["y_min"] <= header_y:
                continue
            t = w["text"].upper()
            if any(k in t for k in ["NET AMOUNT", "BILL TOTAL", "GRAND TOTAL", "TOTAL AMOUNT"]) and w["y_min"] > 700:
                footer_y = w["y_min"]
                # Look ahead for decimal number
                for next_w in words[i:i+6]:
                    num_match = re.search(r"\b(\d+\.\d{2})\b", next_w["text"])
                    if num_match:
                        try:
                            val = float(num_match.group(1))
                            if val > total_amount:
                                total_amount = val
                        except ValueError:
                            pass
                break
        return footer_y, total_amount

    def _cluster_rows_by_y(self, words: List[Dict[str, Any]], y_tolerance: float = 6.5) -> List[List[Dict[str, Any]]]:
        sorted_words = sorted(words, key=lambda w: w["y_mid"])
        rows: List[List[Dict[str, Any]]] = []

        for word in sorted_words:
            matched_row = None
            for row in rows:
                avg_y = sum(w["y_mid"] for w in row) / len(row)
                if abs(word["y_mid"] - avg_y) <= y_tolerance:
                    matched_row = row
                    break
            if matched_row is not None:
                matched_row.append(word)
            else:
                rows.append([word])

        for row in rows:
            row.sort(key=lambda w: w["x_min"])

        return rows

    def _parse_single_row(self, row_words: List[Dict[str, Any]], image_height: int) -> Optional[Dict[str, Any]]:
        row_text = " ".join(w["text"] for w in row_words).strip()
        if not row_text:
            return None

        # Ignore repeated header keywords or table borders
        if any(kw in row_text.upper() for kw in ["HSN", "PRODUCT NAME", "PACK", "BATCH", "EXP.DT", "VALUE SCHEME"]):
            return None
        if re.fullmatch(r"[-=_.\s]+", row_text):
            return None

        y_min = min(w["y_min"] for w in row_words)
        y_max = max(w["y_max"] for w in row_words)
        crop_top = max(0.0, y_min / float(image_height))
        crop_bottom = min(1.0, y_max / float(image_height))

        # Check if this row is a wrapped continuation line (e.g. "NASAL SOLN", "100ML", "SYRUP")
        has_batch = any(self._is_valid_batch(w["text"]) for w in row_words)
        has_expiry = any(re.search(r"\b\d{2}/\d{2,4}\b", w["text"]) for w in row_words)
        has_price = any(re.search(r"\b\d+\.\d{2}\b", w["text"]) for w in row_words)

        if not has_batch and not has_expiry and not has_price:
            # Wrapped description line!
            clean_continuation = re.sub(r"^\d{4,8}\s*", "", row_text).strip()
            if len(clean_continuation) >= 2:
                return {
                    "is_continuation": True,
                    "continuation_text": clean_continuation,
                    "crop_bottom": crop_bottom
                }
            return None

        # Extract tokens for this row
        product_parts = []
        batch_number = ""
        expiry_date = ""
        quantity = 1
        free_quantity = 0
        purchase_rate = 0.0
        mrp = 0.0
        amount = 0.0

        for w in row_words:
            t = w["text"].strip()
            if not t:
                continue

            # Strip leading HSN code e.g. 30049035
            if re.fullmatch(r"\d{8}", t) or (re.fullmatch(r"\d{4,6}", t) and w["x_min"] < 100):
                continue

            # Expiry check: e.g. 04/28 or 04/2028
            exp_match = re.search(r"\b(\d{2}/\d{2,4})\b", t)
            if exp_match and not expiry_date:
                expiry_date = self._normalize_expiry(exp_match.group(1))
                continue

            # Scheme check: e.g. 5+1 or 10+2
            scheme_match = re.fullmatch(r"(\d+)\+(\d+)", t)
            if scheme_match:
                quantity = int(scheme_match.group(1))
                free_quantity = int(scheme_match.group(2))
                continue

            # Decimal numbers: could be Rate, MRP, or Line Amount
            if re.fullmatch(r"\d+\.\d{2}", t):
                val = float(t)
                if purchase_rate == 0.0 and val > 0.0:
                    purchase_rate = val
                elif mrp == 0.0 and val >= purchase_rate:
                    mrp = val
                elif amount == 0.0 and val > 0.0:
                    amount = val
                continue

            # Integer quantity
            if re.fullmatch(r"\d{1,4}", t):
                val = int(t)
                if val > 0 and quantity == 1 and w["x_mid"] > 200:
                    quantity = val
                    continue

            # Batch code: alphanumeric e.g. PJ26019, G2F011108
            if self._is_valid_batch(t) and not batch_number:
                batch_number = t
                continue

            # Pack description (e.g. 10ML, 50GM, 10'S, 100TAB)
            if re.fullmatch(r"\d+('S|GM|ML|N|T|TAB|CAP|KG|BOX)", t, re.IGNORECASE):
                continue

            # Otherwise, it forms part of the product name
            product_parts.append(t)

        product_name = " ".join(product_parts).strip()
        product_name = re.sub(r"^\d{4,8}\s*", "", product_name).strip()

        # Clean product name noise (prefixes, trailing schemes/footers, and extract hidden batch)
        product_name, extra_batch = self._clean_product_name(product_name)
        if not batch_number and extra_batch:
            batch_number = extra_batch

        if not product_name or len(product_name) < 2:
            return None

        # Calculate effective rate/MRP/amount if some were missing
        if purchase_rate > 0.0 and mrp == 0.0:
            mrp = round(purchase_rate * 1.25, 2)
        if mrp > 0.0 and purchase_rate == 0.0:
            purchase_rate = mrp
        if amount == 0.0 and purchase_rate > 0.0:
            amount = round(quantity * purchase_rate, 2)

        # Detect category
        category = self._detect_category(product_name)

        return {
            "is_continuation": False,
            "product_name": product_name,
            "category": category,
            "batch_number": batch_number,
            "expiry_date": expiry_date if expiry_date else "12/2028",
            "quantity": quantity,
            "free_quantity": free_quantity,
            "purchase_rate": purchase_rate,
            "mrp": mrp,
            "amount": amount,
            "crop_top": crop_top,
            "crop_bottom": crop_bottom
        }

    def _clean_product_name(self, name: str) -> Tuple[str, str]:
        extracted_batch = ""
        # 1. Strip page footers anywhere in string
        name = re.sub(r"\bPage\s*\d+\s*(?:of\s*\d+)?\b", "", name, flags=re.IGNORECASE).strip()

        # 2. Strip leading noise chars / symbols / tick letters
        name = re.sub(r"^[\$#\-=_~*•?,\s]+", "", name).strip()
        name = re.sub(r"^\d+[\$sScC]\s*", "", name).strip() # e.g. 20$ or 5c
        name = re.sub(r"^[A-Z][,\s]+(?=[A-Z]{2,})", "", name).strip()
        name = re.sub(r"^(?:IS|T5|300)\s+", "", name).strip()

        # 3. Strip trailing scheme annotations & netrate & symbols
        name = re.sub(r"\b\d+\s*\+\s*\d+\b", "", name).strip()
        name = re.sub(r"\bNETRATE\b", "", name, flags=re.IGNORECASE).strip()
        name = re.sub(r"\s+[\d\.]+$", "", name).strip()
        name = re.sub(r"\s+\d+[\$sScC]$", "", name).strip()
        name = re.sub(r"\b\d+[a-zA-Z]$", "", name).strip()

        # 4. Check for embedded batch at the end of product name
        tokens = name.split()
        if len(tokens) >= 2:
            last = tokens[-1]
            if re.fullmatch(r"[A-Z0-9]{5,12}", last) and (any(c.isdigit() for c in last) or len(last) >= 6):
                if not re.search(r"(?:TAB|CAP|SYP|INJ|GEL|ML|GM|MG)", last, re.IGNORECASE):
                    extracted_batch = last
                    name = " ".join(tokens[:-1]).strip()

        # Final cleanup of trailing hyphens/noise
        name = re.sub(r"[\s\-_?]+$", "", name).strip()
        return name, extracted_batch

    def _is_valid_batch(self, text: str) -> bool:
        if not re.fullmatch(r"[A-Z0-9]{3,14}", text, re.IGNORECASE):
            return False
        # Must have at least one letter and at least one digit or be a clear batch code
        has_letter = any(c.isalpha() for c in text)
        has_digit = any(c.isdigit() for c in text)
        if has_letter and has_digit and len(text) >= 4:
            return True
        # Or standard format like B072
        if text.startswith("B") and len(text) in (4, 5, 6):
            return True
        return False

    def _normalize_expiry(self, exp_str: str) -> str:
        parts = exp_str.split("/")
        if len(parts) == 2:
            m, y = parts[0], parts[1]
            if len(y) == 2:
                y = "20" + y
            return f"{m.zfill(2)}/{y}"
        return exp_str

    def _detect_category(self, name: str) -> str:
        u = name.upper()
        if any(k in u for k in ["TAB", "TABLET", "CAP", "CAPSULE", "SR", "DSR"]):
            return "Medicine"
        if any(k in u for k in ["GEL", "OINT", "OINTMENT", "CREAM", "LOTION"]):
            return "Ointment"
        if any(k in u for k in ["SYP", "SYRUP", "SUSP", "SUSPENSION", "SOLN", "SOLUTION", "DROPS", "DROP"]):
            return "Syrup"
        if any(k in u for k in ["INJ", "INJECTION", "INFUSION", "IV"]):
            return "Injection"
        if any(k in u for k in ["SURGICAL", "GLOVES", "SYRINGE", "NEEDLE", "BANDAGE", "COTTON", "GAUZE", "CATHETER"]):
            return "Surgical"
        return "Medicine"
