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
            if line_data.get("is_continuation"):
                if current_item is not None:
                    current_item["productName"] += " " + line_data.get("continuation_text", "")
                    # Update bottom crop boundary to include wrapped line
                    current_item["cropBottomPercent"] = max(current_item["cropBottomPercent"], line_data.get("crop_bottom", 0.0))
                continue

            if not line_data.get("product_name"):
                continue

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
                "category": line_data.get("category", "Medicine"),
                "batchNumber": line_data.get("batch_number", ""),
                "expiryDate": line_data.get("expiry_date", ""),
                "quantity": line_data.get("quantity", 1),
                "freeQuantity": line_data.get("free_quantity", 0),
                "purchaseRate": line_data.get("purchase_rate", 0.0),
                "mrp": line_data.get("mrp", 0.0),
                "amount": line_data.get("amount", 0.0),
                "cropTopPercent": line_data.get("crop_top", 0.0),
                "cropBottomPercent": line_data.get("crop_bottom", 1.0)
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
        ignore_words = {"NO", "BILL", "INV", "INVOICE", "QTY", "RATE", "MRP", "AMOUNT", "NAME", "SR", "SL", "PATIENT", "DOCTOR", "DR", "CASH", "DATE", "TIME"}
        for i, w in enumerate(words[:40]):
            t = w["text"]
            # Standalone invoice code pattern like F/383149 or INV-2024
            if re.fullmatch(r"[A-Z]{1,4}[/-]\d{4,10}", t):
                return t
            # Check for label "Bill No", "Invoice No", etc.
            if re.search(r"\b(?:Bill\s*No|Invoice\s*No|Inv\s*No|Bill\s*#|Invoice\s*#|Inv\s*#|Invoice\s*:|Bill\s*:)\b", t, re.IGNORECASE):
                parts = re.split(r"[:\s#]+", t)
                for p in reversed(parts):
                    if p.upper() not in ignore_words and len(p) >= 2 and any(ch.isalnum() for ch in p):
                        return p
                if i + 1 < len(words):
                    next_t = words[i + 1]["text"].strip()
                    if next_t.upper() not in ignore_words and len(next_t) >= 2 and any(ch.isalnum() for ch in next_t):
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
        # 1. Normalize comma-separated decimals across all words (e.g. "183,63" -> "183.63")
        for w in row_words:
            w["text"] = re.sub(r"(\d+),(\d{2})\b", r"\1.\2", w["text"])

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

        # Sort row tokens strictly by x_min (left-to-right positional layout)
        row_words.sort(key=lambda w: w["x_min"])

        # 2. Locate Expiry Date token
        expiry_idx = -1
        expiry_date = ""
        for i, w in enumerate(row_words):
            exp_match = re.search(r"\b(\d{2}/\d{2,4})\b", w["text"])
            if exp_match:
                expiry_idx = i
                expiry_date = self._normalize_expiry(exp_match.group(1))
                break

        # 3. Positional Column Extraction for Batch Number
        batch_number = ""
        batch_word_idx = -1
        if expiry_idx > 0:
            # The token directly preceding expiry is candidate batch
            for b_idx in range(expiry_idx - 1, -1, -1):
                candidate_text = row_words[b_idx]["text"].strip()
                # Skip pack tokens e.g. 10ML, 10'S, 100TAB
                if re.fullmatch(r"\d+('S|GM|ML|N|T|TAB|CAP|KG|BOX)", candidate_text, re.IGNORECASE):
                    continue
                if self._is_valid_batch(candidate_text):
                    batch_number = candidate_text
                    batch_word_idx = b_idx
                    break

        # 4. Extract Product Name tokens (all tokens before Batch/Expiry)
        product_parts = []
        name_limit = batch_word_idx if batch_word_idx != -1 else (expiry_idx if expiry_idx != -1 else len(row_words))
        for i in range(name_limit):
            t = row_words[i]["text"].strip()
            if not t:
                continue
            # Strip leading HSN code e.g. 30049035
            if i == 0 and (re.fullmatch(r"\d{8}", t) or (re.fullmatch(r"\d{4,6}", t) and row_words[i]["x_min"] < 120)):
                continue
            # Skip pack tokens
            if re.fullmatch(r"\d+('S|GM|ML|N|T|TAB|CAP|KG|BOX)", t, re.IGNORECASE):
                continue
            product_parts.append(t)

        # 5. Extract Numeric Values (all tokens after Expiry)
        quantity = 1
        free_quantity = 0
        purchase_rate = 0.0
        mrp = 0.0
        amount = 0.0
        decimals: List[float] = []

        numeric_start = (expiry_idx + 1) if expiry_idx != -1 else 0
        for i in range(numeric_start, len(row_words)):
            t = row_words[i]["text"].strip()
            if not t:
                continue

            # Scheme check: e.g. 5+1 or 10+2
            scheme_match = re.fullmatch(r"(\d+)\+(\d+)", t)
            if scheme_match:
                quantity = int(scheme_match.group(1))
                free_quantity = int(scheme_match.group(2))
                continue

            # Decimals: Rate, MRP, Disc%, Amount
            if re.fullmatch(r"\d+\.\d+", t):
                try:
                    decimals.append(float(t))
                except ValueError:
                    pass
                continue

            # Integer quantity (if not already set by scheme)
            if re.fullmatch(r"\d{1,4}", t):
                val = int(t)
                if val > 0 and quantity == 1 and free_quantity == 0:
                    quantity = val
                    continue

        # Reconcile Decimals mathematically
        if len(decimals) == 1:
            purchase_rate = decimals[0]
            mrp = round(purchase_rate * 1.25, 2)
            amount = round(quantity * purchase_rate, 2)
        elif len(decimals) == 2:
            d1, d2 = decimals[0], decimals[1]
            purchase_rate = min(d1, d2)
            mrp = max(d1, d2)
            amount = round(quantity * purchase_rate, 2)
        elif len(decimals) >= 3:
            # Typical order: [Rate, MRP, (Disc%), (Tax%), Amount]
            d_first = decimals[0]
            d_second = decimals[1]
            purchase_rate = d_first
            mrp = d_second if d_second >= purchase_rate else round(purchase_rate * 1.25, 2)

            expected_line_amount = round(quantity * purchase_rate, 2)
            # Find candidate among remaining decimals that best matches expected line amount
            remaining = decimals[2:]
            best_match = None
            for d in remaining:
                if abs(d - expected_line_amount) <= max(1.0, expected_line_amount * 0.15):
                    best_match = d
                    break

            if best_match is not None:
                amount = best_match
            elif remaining and remaining[-1] > 5.0 and remaining[-1] >= purchase_rate:
                amount = remaining[-1]
            else:
                amount = expected_line_amount

        # Final sanity check on math: Amount must never be a fraction of Qty * Rate
        expected_min = round(quantity * purchase_rate * 0.5, 2)
        if amount <= 0.0 or (purchase_rate > 0.0 and amount < expected_min):
            amount = round(quantity * purchase_rate, 2)

        if mrp <= 0.0 and purchase_rate > 0.0:
            mrp = round(purchase_rate * 1.25, 2)

        # 6. Clean product name
        product_name = " ".join(product_parts).strip()
        product_name, extra_batch = self._clean_product_name(product_name)
        if not batch_number and extra_batch:
            batch_number = extra_batch

        if not product_name or len(product_name) < 2:
            return None

        # Reject product names that are purely digits or punctuation (e.g. dates, numbers)
        alpha_chars = [c for c in product_name if c.isalpha()]
        if len(alpha_chars) < 2:
            return None

        # Reject header/footer lines or phone numbers that bled into row detection
        upper_name = product_name.upper()
        header_keywords = [
            "CASH BILL", "DL NO", "MOB:", "MOBILE", "PHONE", "TOTAL",
            "AMOUNT", "NAME OF", "PRODUCTS", "PATIENT", "DOCTOR", "DR.",
            "SUB TOTAL", "NET AMOUNT", "GRAND TOTAL"
        ]
        if any(kw in upper_name for kw in header_keywords) and len(alpha_chars) < 6:
            return None
        if re.search(r"\b[6-9]\d{9}\b", product_name):
            return None

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

        # 2. Strip leading HSN codes (8-digit numbers or OCR-corrupted HSN like 30D4907S, 300420TS, O04909S)
        name = re.sub(r"^(?:30[0-9A-Z]{4,7}|[O0][0-9A-Z]{6,8}|\d{4,8})\s+", "", name).strip()
        name = re.sub(r"^(?:IS|T5|300|iie|Z5|ZOC|D9C|2OC\d*)\s*[-:]?\s*", "", name, flags=re.IGNORECASE).strip()
        name = re.sub(r"^[A-Z][,\s]+(?=[A-Z]{2,})", "", name).strip()
        name = re.sub(r"^[\$#\-=_~*•?,\s]+", "", name).strip()
        name = re.sub(r"^\d+[\$sScC]\s*", "", name).strip()

        # 3. Strip trailing prices, decimals, rates, noise
        name = re.sub(r"\b\d+\s*\+\s*\d+\b", "", name).strip() # schemes
        name = re.sub(r"\bNETRATE\b", "", name, flags=re.IGNORECASE).strip()
        name = re.sub(r"\s+[\d\.,]+(?:\s*/\s*\d+)?\.?$", "", name).strip() # trailing prices like "299.64 /23." or "183.63"
        name = re.sub(r"\s+[0-9]+[L|I|l]\d+(?:\.\d+)?$", "", name).strip() # trailing garbage like "3L19.50"
        name = re.sub(r"\s+\d+[\$sScC]$", "", name).strip()
        name = re.sub(r"\b\d+[a-zA-Z]$", "", name).strip()

        # 4. Check for embedded batch at the end of product name
        tokens = name.split()
        if len(tokens) >= 2:
            last = tokens[-1]
            if self._is_valid_batch(last) and not re.search(r"(?:TAB|CAP|SYP|INJ|GEL|ML|GM|MG)", last, re.IGNORECASE):
                extracted_batch = last
                name = " ".join(tokens[:-1]).strip()

        # Final cleanup of trailing hyphens/noise
        name = re.sub(r"[\s\-_?]+$", "", name).strip()
        return name, extracted_batch

    def _is_valid_batch(self, text: str) -> bool:
        if not re.fullmatch(r"[A-Z0-9]{3,14}", text, re.IGNORECASE):
            return False
        # 1. Pure numeric batch (5 to 10 digits e.g. 7260711, 240105)
        if text.isdigit() and 5 <= len(text) <= 10:
            return True
        # 2. Alphanumeric batch with at least one letter and one digit
        has_letter = any(c.isalpha() for c in text)
        has_digit = any(c.isdigit() for c in text)
        if has_letter and has_digit and len(text) >= 4:
            return True
        # 3. Standard format starting with B / G / K / L / P / F
        if text[0].upper() in "BGKLPF" and 4 <= len(text) <= 10:
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

    def prepare_words(
        self,
        texts: List[str],
        scores: List[float],
        boxes: List[List[float]]
    ) -> List[Dict[str, Any]]:
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
        return words

    def get_row_clusters_and_header(
        self,
        words: List[Dict[str, Any]]
    ) -> Tuple[List[str], str, List[List[Dict[str, Any]]], str, str, str, float]:
        """
        Extracts header information, clusters body text into physical lines,
        and returns data formatted for LLM structuring.
        """
        supplier_name = self._extract_supplier_name(words)
        invoice_no = self._extract_invoice_number(words)
        bill_date = self._extract_bill_date(words)
        table_header_y = self._find_table_header_y(words)
        footer_y, total_amount = self._find_footer_y_and_total(words, table_header_y)

        body_words = [
            w for w in words
            if w["y_min"] >= table_header_y and (footer_y == 0 or w["y_min"] < footer_y + 15)
        ]
        rows = self._cluster_rows_by_y(body_words, y_tolerance=9.5)

        clustered_lines = []
        for r in rows:
            # Sort words in row from left to right
            sorted_w = sorted(r, key=lambda w: w["x_min"])
            line_str = " | ".join(w["text"] for w in sorted_w)
            if line_str.strip():
                clustered_lines.append(line_str)

        header_summary = f"Supplier: {supplier_name}, Invoice #: {invoice_no}, Date: {bill_date}, Detected Total: ₹{total_amount}"
        return clustered_lines, header_summary, rows, supplier_name, invoice_no, bill_date, total_amount

    def reconcile_llm_result(
        self,
        llm_data: Dict[str, Any],
        rows: List[List[Dict[str, Any]]],
        image_height: int,
        fallback_supplier: str = "",
        fallback_invoice: str = "",
        fallback_date: str = "",
        fallback_total: float = 0.0
    ) -> Optional[OcrDraftBillResponse]:
        """
        Reconciles Qwen LLM output with physical row bounding boxes and strict arithmetic integrity.
        """
        raw_items = llm_data.get("items")
        if not raw_items or not isinstance(raw_items, list):
            return None

        supplier_name = (llm_data.get("supplierName") or fallback_supplier).strip()
        invoice_no = (llm_data.get("invoiceNo") or fallback_invoice).strip()
        bill_date = (llm_data.get("billDate") or fallback_date).strip()

        reconciled_items: List[OcrDraftItemResponse] = []
        for i, item in enumerate(raw_items):
            if not isinstance(item, dict):
                continue
            name = str(item.get("productName", "")).strip()
            if not name or len(name) < 2:
                continue

            # Clean name noise
            cleaned_name, ext_batch = self._clean_product_name(name)
            name = cleaned_name
            if not name:
                continue

            batch = str(item.get("batchNumber", "")).strip()
            if not batch and ext_batch:
                batch = ext_batch
            expiry = str(item.get("expiryDate", "")).strip()
            if expiry:
                expiry = self._normalize_expiry(expiry)

            try:
                qty = max(1, int(float(item.get("quantity", 1))))
            except Exception:
                qty = 1

            try:
                free_qty = max(0, int(float(item.get("freeQuantity", 0))))
            except Exception:
                free_qty = 0

            try:
                rate = round(float(item.get("purchaseRate", 0.0)), 2)
            except Exception:
                rate = 0.0

            try:
                mrp = round(float(item.get("mrp", 0.0)), 2)
            except Exception:
                mrp = 0.0

            # Default MRP if rate is known
            if mrp <= 0.0 and rate > 0.0:
                mrp = round(rate * 1.25, 2)
            elif rate <= 0.0 and mrp > 0.0:
                rate = mrp

            # Strict arithmetic verification
            amount = round(qty * rate, 2)

            # Assign row crop bounds by approximate row index or matching
            row_idx = min(i, len(rows) - 1) if rows else 0
            if rows and row_idx < len(rows):
                target_row = rows[row_idx]
                min_y = min(w["y_min"] for w in target_row)
                max_y = max(w["y_max"] for w in target_row)
                crop_top = max(0.0, (min_y - 4.0) / max(1, image_height))
                crop_bottom = min(1.0, (max_y + 4.0) / max(1, image_height))
            else:
                crop_top = 0.0
                crop_bottom = 1.0

            reconciled_items.append(OcrDraftItemResponse(
                physicalRowId=f"ROW_{i+1:02d}",
                productName=name,
                category=self._detect_category(name),
                batchNumber=batch,
                expiryDate=expiry,
                quantity=qty,
                freeQuantity=free_qty,
                purchaseRate=rate,
                mrp=mrp,
                amount=amount,
                cropTopPercent=crop_top,
                cropBottomPercent=crop_bottom
            ))

        if not reconciled_items:
            return None

        total_amount = round(sum(it.amount for it in reconciled_items), 2)
        if fallback_total > 0 and abs(fallback_total - total_amount) < 2.0:
            total_amount = fallback_total

        return OcrDraftBillResponse(
            supplierName=supplier_name if supplier_name else "Pharmaceutical Supplier",
            invoiceNo=invoice_no if invoice_no else "",
            billDate=bill_date if bill_date else "",
            totalAmount=total_amount,
            parsedSource="PaddleOCR + Qwen Hybrid Engine",
            items=reconciled_items
        )
