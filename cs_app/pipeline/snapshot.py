import logging
from pathlib import Path

import cv2
import numpy as np

from cs_app.config import OBSERVATION_IMAGES_DIR

logger = logging.getLogger(__name__)

_PADDING = 10  # pixels of extra context around the bounding box


def save_snapshot(frame_bgr: np.ndarray, bbox_xyxy: list[float], observation_id: str) -> Path:
    """
    Crop the vehicle bounding box (with padding) and save as JPEG.
    Returns the saved file path.
    """
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox_xyxy)
    x1 = max(0, x1 - _PADDING)
    y1 = max(0, y1 - _PADDING)
    x2 = min(w, x2 + _PADDING)
    y2 = min(h, y2 + _PADDING)

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        logger.warning("Empty crop for %s — skipping snapshot", observation_id)
        return None

    out_path = OBSERVATION_IMAGES_DIR / f"{observation_id}.jpg"
    cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out_path
