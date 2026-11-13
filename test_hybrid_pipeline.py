import unittest
from app.table_reconstructor import TableReconstructor

class TestHybridPipeline(unittest.TestCase):
    def setUp(self):
        self.reconstructor = TableReconstructor()

    def test_reconcile_llm_items_exact_math_and_numeric_batch(self):
        mock_llm_data = {
            "supplierName": "CENTAUR PHARMACEUTICALS PVT LTD",
            "invoiceNo": "INV-78921",
            "billDate": "12/11/2026",
            "items": [
                {
                    "productName": "SINAREST NEW TAB",
                    "batchNumber": "7260711",
                    "expiryDate": "11/2026",
                    "quantity": 10,
                    "freeQuantity": 2,
                    "purchaseRate": 183.63,
                    "mrp": 245.00
                },
                {
                    "productName": "AUGMENTIN 625 DUO",
                    "batchNumber": "B9901",
                    "expiryDate": "05/2028",
                    "quantity": 5,
                    "freeQuantity": 0,
                    "purchaseRate": 150.50,
                    "mrp": 200.00
                }
            ]
        }

        # Mock rows with Y coordinates
        mock_rows = [
            [{"y_min": 100, "y_max": 120, "x_min": 10, "x_max": 200, "text": "SINAREST NEW TAB"}],
            [{"y_min": 130, "y_max": 150, "x_min": 10, "x_max": 200, "text": "AUGMENTIN 625 DUO"}]
        ]

        result = self.reconstructor.reconcile_llm_result(
            llm_data=mock_llm_data,
            rows=mock_rows,
            image_height=1000
        )

        self.assertNotNull = self.assertIsNotNone
        self.assertIsNotNone(result)
        self.assertEqual("PaddleOCR + Qwen Hybrid Engine", result.parsedSource)
        self.assertEqual("CENTAUR PHARMACEUTICALS PVT LTD", result.supplierName)
        self.assertEqual(2, len(result.items))

        item1 = result.items[0]
        self.assertEqual("SINAREST NEW TAB", item1.productName)
        self.assertEqual("7260711", item1.batchNumber)
        self.assertEqual("11/2026", item1.expiryDate)
        self.assertEqual(10, item1.quantity)
        self.assertEqual(2, item1.freeQuantity)
        self.assertEqual(183.63, item1.purchaseRate)
        # Math must be strictly 10 * 183.63 = 1836.30
        self.assertEqual(1836.30, item1.amount)

        item2 = result.items[1]
        self.assertEqual("AUGMENTIN 625 DUO", item2.productName)
        self.assertEqual("B9901", item2.batchNumber)
        self.assertEqual(5, item2.quantity)
        self.assertEqual(150.50, item2.purchaseRate)
        # Math must be strictly 5 * 150.50 = 752.50
        self.assertEqual(752.50, item2.amount)

        # Total amount must be exact sum of lines
        expected_total = round(1836.30 + 752.50, 2)
        self.assertEqual(expected_total, result.totalAmount)

    def test_reconcile_invalid_items_returns_none_for_fallback(self):
        # Empty or invalid items should trigger None so native reconstructor kicks in
        result = self.reconstructor.reconcile_llm_result(
            llm_data={"items": []},
            rows=[],
            image_height=1000
        )
        self.assertIsNone(result)

    def test_reconstruct_continuation_without_prior_item_does_not_crash(self):
        # Header + continuation row (no previous item) - previously threw KeyError: 'product_name'
        texts = ["M/s. SIYARAM MEDICAL", "Name of the Products", "Qty.", "Amount", "NASAL DROPS ONLY", "Total :"]
        scores = [0.95, 0.9, 0.9, 0.9, 0.9, 0.9]
        boxes = [
            [20, 20, 200, 40],
            [20, 160, 150, 180],
            [160, 160, 200, 180],
            [210, 160, 260, 180],
            [20, 220, 150, 240], # continuation text in body
            [20, 750, 100, 780]
        ]
        result = self.reconstructor.reconstruct_invoice(
            texts=texts,
            scores=scores,
            boxes=boxes,
            image_height=1000,
            image_width=800
        )
        self.assertIsNotNone(result)
        self.assertEqual("M/s. SIYARAM MEDICAL", result.supplierName)
        # Should gracefully return empty items list without raising KeyError
        self.assertEqual(0, len(result.items))

if __name__ == "__main__":
    unittest.main()
