import asyncio
import os
import json
from app.main import parse_bill
from fastapi import UploadFile

async def run_detailed_test():
    image_path = "/Users/kalpakkishornarkhede/.gemini/antigravity/brain/59866d4f-2cb3-4fa9-89b0-a5e19f32c20a/.user_uploaded/media_1794383940505_da466d55.jpg"
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    with open(image_path, "rb") as f:
        upload_file = UploadFile(filename="pharmacy_bill_test.jpg", file=f)
        result = await parse_bill(upload_file)

    print("=" * 80)
    print("                    PADDLEOCR v5 DETAILED EXTRACTION REPORT")
    print("=" * 80)
    print(f"Distributor / Supplier : {result.supplierName}")
    print(f"Invoice Number         : {result.invoiceNo}")
    print(f"Invoice Date           : {result.billDate}")
    print(f"Total Printed Amount   : ₹{result.totalAmount}")
    print(f"OCR Pipeline Engine    : {result.parsedSource}")
    print(f"Total Items Extracted  : {len(result.items)} items")
    print("=" * 80)
    print(f"{'#':<3} | {'Product Name':<32} | {'Batch':<10} | {'Exp':<8} | {'Qty':<4} | {'Free':<4} | {'Rate (₹)':<8} | {'MRP (₹)':<8} | {'Amount (₹)':<10}")
    print("-" * 80)

    calculated_total = 0.0
    for i, it in enumerate(result.items, 1):
        line_calc = round(it.quantity * it.purchaseRate, 2)
        calculated_total += it.amount
        print(f"{i:<3} | {it.productName[:32]:<32} | {it.batchNumber:<10} | {it.expiryDate:<8} | {it.quantity:<4} | {it.freeQuantity:<4} | {it.purchaseRate:<8.2f} | {it.mrp:<8.2f} | {it.amount:<10.2f}")

    print("-" * 80)
    print(f"Sum of extracted item amounts : ₹{calculated_total:.2f}")
    print(f"Printed invoice grand total   : ₹{result.totalAmount:.2f}")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(run_detailed_test())
