import cv2
import numpy as np


# HSV hue ranges (OpenCV: H 0-179, S 0-255, V 0-255)
_COLOR_RANGES = [
    ("red",    [(0, 30, 50), (10, 255, 255)]),
    ("red",    [(165, 30, 50), (179, 255, 255)]),   # red wraps around hue 0
    ("orange", [(11, 80, 80), (25, 255, 255)]),
    ("yellow", [(26, 80, 80), (34, 255, 255)]),
    ("green",  [(35, 40, 40), (85, 255, 255)]),
    ("blue",   [(86, 40, 40), (130, 255, 255)]),
    ("purple", [(131, 30, 30), (164, 255, 255)]),
    ("white",  [(0, 0, 180), (179, 30, 255)]),
    ("silver", [(0, 0, 120), (179, 30, 179)]),
    ("black",  [(0, 0, 0),   (179, 255, 50)]),
]


def classify_vehicle_color(frame_bgr: np.ndarray, bbox_xyxy: list[float]) -> str:
    """
    Estimate the dominant body color of a vehicle from its bounding box crop.
    Returns a color name string.
    """
    x1, y1, x2, y2 = (int(v) for v in bbox_xyxy)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return "unknown"

    # Focus on the central 60% of the crop (avoids road/sky contamination)
    h, w = crop.shape[:2]
    cy1, cy2 = int(h * 0.2), int(h * 0.8)
    cx1, cx2 = int(w * 0.1), int(w * 0.9)
    roi = crop[cy1:cy2, cx1:cx2]
    if roi.size == 0:
        roi = crop

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    best_color = "grey"
    best_count = 0

    for color_name, (lower, upper) in _COLOR_RANGES:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        count = int(np.sum(mask > 0))
        if count > best_count:
            best_count = count
            best_color = color_name

    return best_color
