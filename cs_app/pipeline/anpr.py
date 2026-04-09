import logging
import re
from dataclasses import dataclass

import numpy as np
import easyocr

from cs_app.config import ANPR_CONFIDENCE_THRESHOLD
from cs_app.pipeline.plate_color import classify_plate_color, extract_plate_region

logger = logging.getLogger(__name__)


@dataclass
class ANPRResult:
    text: str
    confidence: float
    plate_color: str
    category: str
    nationality: str


# Regex patterns for common European plate formats → country code
_NATIONALITY_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^[A-Z]{2}\d{2}[A-Z]{3}$"), "GB"),
    (re.compile(r"^[A-Z]{3}\d{3}$"), "SE"),
    (re.compile(r"^[A-Z]{2}\d{5}$"), "NO"),
    (re.compile(r"^[A-Z]{2}\d{2}\d{3}$"), "DK"),
    (re.compile(r"^[A-Z]{1,3}[A-Z]{1,2}\d{1,4}[HE]?$"), "DE"),
    (re.compile(r"^[A-Z]{2}\d{3}[A-Z]{2}$"), "FR"),
    (re.compile(r"^[A-Z]{2}\d{3}[A-Z]$"), "NL"),
    (re.compile(r"^[A-Z]{2,3}\d{4,5}$"), "PL"),
    (re.compile(r"^[A-Z]{2,3}\d{3}$"), "FI"),
    (re.compile(r"^[A-Z]{1,3}\d{3,4}[A-Z]{0,3}$"), "Unknown"),
]

_CLEAN_RE = re.compile(r"[\s\-\.]")


class ANPRProcessor:
    def __init__(self):
        logger.info("Initialising EasyOCR (gpu=True) ...")
        self.reader = easyocr.Reader(["en"], gpu=True)
        logger.info("EasyOCR ready.")

    def read_plate(
        self, frame_bgr: np.ndarray, bbox_xyxy: list[float]
    ) -> ANPRResult:
        """
        Read the licence plate from a vehicle bounding box.

        Plate background color is classified from the tight region around the
        OCR-detected text — not from the vehicle body — so car color cannot
        influence taxi/diplomatic categorisation.
        """
        plate_crop = extract_plate_region(frame_bgr, bbox_xyxy)

        if plate_crop.size == 0:
            return ANPRResult("", 0.0, "white", "normal", "Unknown")

        try:
            ocr_results = self.reader.readtext(plate_crop, detail=1)
        except Exception as e:
            logger.warning("EasyOCR failed: %s", e)
            return ANPRResult("", 0.0, "white", "normal", "Unknown")

        best_text = ""
        best_conf = 0.0
        best_ocr_bbox = None  # tight bbox around the best OCR result (in plate_crop coords)

        for (ocr_bbox, text, conf) in ocr_results:
            clean = _CLEAN_RE.sub("", text.upper())
            if not re.match(r"^[A-Z0-9]{4,10}$", clean):
                continue
            if conf > best_conf:
                best_conf = conf
                best_text = clean
                best_ocr_bbox = ocr_bbox  # list of 4 [x,y] corner points

        # Classify plate background color from the tight OCR bbox, not the whole
        # bottom-strip crop. This ensures car body colour cannot affect the result.
        plate_color, category = _classify_from_ocr_bbox(plate_crop, best_ocr_bbox)

        nationality = infer_nationality(best_text) if best_text else "Unknown"

        return ANPRResult(
            text=best_text,
            confidence=best_conf,
            plate_color=plate_color,
            category=category,
            nationality=nationality,
        )

    def should_retry(self, result: ANPRResult) -> bool:
        """True if plate confidence is below the lock threshold."""
        return not result.text or result.confidence < ANPR_CONFIDENCE_THRESHOLD


def _classify_from_ocr_bbox(
    plate_crop: np.ndarray,
    ocr_bbox,
) -> tuple[str, str]:
    """
    If an OCR bbox is available, crop tightly around the detected text and
    classify the plate background from that region.
    Falls back to the full plate_crop if no OCR bbox is available.
    """
    if ocr_bbox is not None:
        try:
            xs = [pt[0] for pt in ocr_bbox]
            ys = [pt[1] for pt in ocr_bbox]
            h, w = plate_crop.shape[:2]
            # Add a small margin around the text to include the plate background
            margin = max(4, int((max(ys) - min(ys)) * 0.4))
            x1 = max(0, int(min(xs)) - margin)
            y1 = max(0, int(min(ys)) - margin)
            x2 = min(w, int(max(xs)) + margin)
            y2 = min(h, int(max(ys)) + margin)
            tight_crop = plate_crop[y1:y2, x1:x2]
            if tight_crop.size > 0:
                return classify_plate_color(tight_crop)
        except Exception:
            pass

    return classify_plate_color(plate_crop)


def infer_nationality(plate_text: str) -> str:
    clean = _CLEAN_RE.sub("", plate_text.upper())
    for pattern, country in _NATIONALITY_RULES:
        if pattern.match(clean):
            return country
    return "Unknown"
