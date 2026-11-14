import os
import re
import json
import time
import httpx
from typing import List, Dict, Any, Optional

class QwenInvoiceParser:
    """
    Intelligent LLM-based pharmaceutical invoice parser using Qwen models.
    Supports Hugging Face Serverless Inference API, OpenAI-compatible APIs (Groq/OpenRouter/DashScope),
    and local Ollama with a strict timeout and graceful fallback to native spatial OCR.
    """

    def __init__(self):
        self.hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
        self.qwen_api_key = os.getenv("QWEN_API_KEY") or self.hf_token
        self.custom_endpoint = os.getenv("QWEN_BASE_URL")
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.model_name = os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        self.timeout_seconds = float(os.getenv("QWEN_TIMEOUT_SECONDS", "3.0"))

    def is_configured(self) -> bool:
        """Checks if a remote API key/token is present or if Ollama URL is configured."""
        return bool(self.hf_token or self.qwen_api_key or os.getenv("ENABLE_OLLAMA_PARSER"))

    async def parse_ocr_text_with_qwen(
        self,
        clustered_lines: List[str],
        header_text: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Submits OCR lines to Qwen to extract clean structured pharmaceutical invoice data.
        Returns parsed dictionary or None if timeout / error occurs.
        """
        if not clustered_lines:
            return None

        prompt = self._build_prompt(clustered_lines, header_text)

        # 1. Try Hugging Face Serverless / OpenAI-compatible endpoint
        if self.qwen_api_key or self.hf_token:
            result = await self._call_hf_or_openai_endpoint(prompt)
            if result:
                return result

        # 2. Try Ollama if configured
        if os.getenv("ENABLE_OLLAMA_PARSER"):
            result = await self._call_ollama_endpoint(prompt)
            if result:
                return result

        return None

    def _build_prompt(self, lines: List[str], header_text: str) -> str:
        formatted_lines = "\n".join([f"{i+1}: {line}" for i, line in enumerate(lines)])
        return f"""You are a pharmaceutical invoice data structuring specialist.
Extract invoiced medicine items, batch codes, expiry dates, quantities, and pricing from the following OCR-extracted bill lines.

HEADER INFO DETECTED:
{header_text if header_text else "Not specified"}

OCR LINES:
{formatted_lines}

RULES:
1. Extract ALL legitimate pharmaceutical products / surgicals / medicines.
2. Maintain exact pure numeric (e.g. 7260711) or alphanumeric batch numbers. Do NOT merge batch codes into product names.
3. Expiry format must be MM/YYYY or MM/YY.
4. Normalize comma decimals (e.g. 183,63 -> 183.63).
5. Purchase Rate is unit wholesale cost. MRP is retail selling price. INVARIANT: Purchase Rate MUST ALWAYS be less than or equal to MRP (purchaseRate <= mrp). Never invert them.
6. Return ONLY valid JSON in this exact structure with no conversational text:

{{
  "supplierName": "Supplier Name or Distributors",
  "invoiceNo": "Invoice/Bill Number",
  "billDate": "DD/MM/YYYY",
  "items": [
    {{
      "productName": "SINAREST NEW TAB",
      "batchNumber": "7260711",
      "expiryDate": "11/2026",
      "quantity": 10,
      "freeQuantity": 0,
      "purchaseRate": 183.63,
      "mrp": 245.00
    }}
  ]
}}"""

    async def _call_hf_or_openai_endpoint(self, prompt: str) -> Optional[Dict[str, Any]]:
        # Hugging Face Serverless Inference endpoint or custom OpenAI endpoint
        endpoint = self.custom_endpoint or f"https://api-inference.huggingface.co/models/{self.model_name}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.qwen_api_key or self.hf_token}"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a precise pharmaceutical invoice data extraction engine. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 1500
        }

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_seconds, connect=1.0)) as client:
                resp = await client.post(endpoint, headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    return self._extract_json_from_text(content)
                else:
                    print(f"[QwenParser] HF/OpenAI endpoint returned {resp.status_code}: {resp.text[:200]}")
                    return None
        except httpx.TimeoutException:
            print(f"[QwenParser] Remote Qwen call timed out after {self.timeout_seconds}s. Falling back to native spatial reconstructor.")
            return None
        except Exception as e:
            print(f"[QwenParser] Remote Qwen call failed: {e}. Falling back to native spatial reconstructor.")
            return None

    async def _call_ollama_endpoint(self, prompt: str) -> Optional[Dict[str, Any]]:
        url = f"{self.ollama_url.rstrip('/')}/api/generate"
        payload = {
            "model": "qwen2.5:7b",
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_seconds, connect=0.5)) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    return self._extract_json_from_text(data.get("response", ""))
                return None
        except Exception as e:
            print(f"[QwenParser] Ollama call failed: {e}")
            return None

    def _sanitize_extracted_json(self, data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not data or not isinstance(data, dict):
            return data
        items = data.get("items")
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    pr = float(item.get("purchaseRate", 0.0) or 0.0)
                    mrp = float(item.get("mrp", 0.0) or 0.0)
                    if pr > mrp and mrp > 0.0:
                        item["purchaseRate"] = mrp
                        item["mrp"] = pr
                    elif mrp <= 0.0 and pr > 0.0:
                        item["mrp"] = round(pr * 1.25, 2)
        return data

    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        parsed = None
        try:
            # 1. Direct JSON parse
            parsed = json.loads(text.strip())
        except Exception:
            pass

        # 2. Extract from ```json ... ``` code fence
        if not parsed:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if match:
                try:
                    parsed = json.loads(match.group(1).strip())
                except Exception:
                    pass

        # 3. Search for outermost balanced { ... }
        if not parsed:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(text[start:end+1])
                except Exception:
                    pass

        return self._sanitize_extracted_json(parsed)
