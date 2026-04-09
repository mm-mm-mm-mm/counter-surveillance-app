import cv2
import numpy as np


def classify_plate_color(plate_crop_bgr: np.ndarray) -> tuple[str, str]:
    """
    Analyse the background color of a licence plate crop.

    Returns:
        (plate_color_name, category)
        where category is one of: "normal", "taxi", "diplomatic"
    """
    if plate_crop_bgr is None or plate_crop_bgr.size == 0:
        return "white", "normal"

    hsv = cv2.cvtColor(plate_crop_bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]

    # Sample background pixels from the four corners (avoids text)
    margin = max(1, min(h // 4, w // 4, 8))
    corners = np.vstack([
        hsv[:margin, :margin].reshape(-1, 3),
        hsv[:margin, -margin:].reshape(-1, 3),
        hsv[-margin:, :margin].reshape(-1, 3),
        hsv[-margin:, -margin:].reshape(-1, 3),
    ])

    if len(corners) == 0:
        return "white", "normal"

    # Use median to be robust against stray pixels
    h_med = float(np.median(corners[:, 0]))
    s_med = float(np.median(corners[:, 1]))
    v_med = float(np.median(corners[:, 2]))

    # Blue: diplomatic plates
    if 95 <= h_med <= 135 and s_med > 70:
        return "blue", "diplomatic"

    # Yellow: taxi plates
    if 18 <= h_med <= 38 and s_med > 90:
        return "yellow", "taxi"

    # White/light: normal plates
    if s_med < 50 and v_med > 150:
        return "white", "normal"

    # Default
    return "white", "normal"


def extract_plate_region(frame_bgr: np.ndarray, bbox_xyxy: list[float]) -> np.ndarray:
    """
    Heuristic: the licence plate is usually in the bottom 35% of the vehicle bbox.
    Returns the cropped region as a BGR array.
    """
    x1, y1, x2, y2 = (int(v) for v in bbox_xyxy)
    plate_y1 = y1 + int((y2 - y1) * 0.65)
    crop = frame_bgr[plate_y1:y2, x1:x2]
    return crop
