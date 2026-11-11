import cv2
import numpy as np
from typing import Tuple, Dict, Any

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 points as: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def preprocess_and_correct_perspective(image_bytes: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Decodes image, detects document quadrilateral boundary if present,
    and applies OpenCV perspective transform. Returns corrected BGR image.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image from provided bytes.")

    orig_h, orig_w = image.shape[:2]
    meta = {
        "original_width": orig_w,
        "original_height": orig_h,
        "perspective_corrected": False
    }

    # Downscale for fast contour detection
    max_dim = 1200
    scale = 1.0
    if max(orig_h, orig_w) > max_dim:
        scale = max_dim / float(max(orig_h, orig_w))
        resized = cv2.resize(image, (int(orig_w * scale), int(orig_h * scale)))
    else:
        resized = image.copy()

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)

    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    doc_cnt = None
    min_area = (resized.shape[0] * resized.shape[1]) * 0.25

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > min_area:
            doc_cnt = approx
            break

    if doc_cnt is not None:
        pts = doc_cnt.reshape(4, 2) / scale
        rect = order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        meta["perspective_corrected"] = True
        meta["corrected_width"] = maxWidth
        meta["corrected_height"] = maxHeight
        return warped, meta

    # If no 4-point quadrilateral found (e.g., invoice already cropped or close up), return original image
    return image, meta
