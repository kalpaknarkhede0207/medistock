import asyncio
from app.main import parse_bill
from fastapi import UploadFile

async def test_end_to_end():
    image_path = "/Users/kalpakkishornarkhede/.gemini/antigravity/brain/59866d4f-2cb3-4fa9-89b0-a5e19f32c20a/.user_uploaded/media_1794383940505_da466d55.jpg"
    with open(image_path, "rb") as f:
        content = f.read()

    upload_file = UploadFile(filename="test_invoice.jpg", file=open(image_path, "rb"))
    result = await parse_bill(upload_file)

    print("=" * 60)
    print("PADDLEOCR BACKEND EXTRACTION RESULTS:")
    print("=" * 60)
    print(f"Supplier Name: {result.supplierName}")
    print(f"Invoice No   : {result.invoiceNo}")
    print(f"Bill Date    : {result.billDate}")
    print(f"Total Amount : ₹{result.totalAmount}")
    print(f"Source Engine: {result.parsedSource}")
    print(f"Total Items  : {len(result.items)}")
    print("-" * 60)

    for item in result.items:
        print(f"[{item.physicalRowId}] {item.productName}")
        print(f"   Batch: {item.batchNumber} | Exp: {item.expiryDate} | Qty: {item.quantity} (Free: {item.freeQuantity}) | Rate: ₹{item.purchaseRate} | MRP: ₹{item.mrp} | Amount: ₹{item.amount}")

    print("=" * 60)
    assert len(result.items) > 0, "No items extracted!"
    assert result.supplierName != "", "Supplier name missing!"
    print("End-to-end verification PASSED!")

if __name__ == "__main__":
    asyncio.run(test_end_to_end())
