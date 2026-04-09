import logging
import re
from dataclasses import dataclass

import numpy as np
import easyocr

from cs_app.config import ANPR_CONFIDENCE_THRESHOLD
from cs_app.pipeline.plate_color import extract_plate_region

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

# Accepted plate formats (L = letter, N = number):
#   LLL NNN  →  3 letters + 3 digits      (e.g. ABC123)
#   LLL NN L →  3 letters + 2 digits + 1 letter  (e.g. ABC12D)
#   LL NNN L →  2 letters + 3 digits + 1 letter  (e.g. AB123C)
_VALID_PLATE_RE = re.compile(
    r"^(?:[A-Z]{3}\d{3}|[A-Z]{3}\d{2}[A-Z]|[A-Z]{2}\d{3}[A-Z])$"
)


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

        for (ocr_bbox, text, conf) in ocr_results:
            clean = _CLEAN_RE.sub("", text.upper())
            if not _is_valid_plate(clean):
                continue
            if conf > best_conf:
                best_conf = conf
                best_text = clean

        # Category classification from plate background colour is disabled —
        # all vehicles default to "normal".
        plate_color, category = "white", "normal"

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


def _is_valid_plate(text: str) -> bool:
    return bool(_VALID_PLATE_RE.match(text))


def infer_nationality(plate_text: str) -> str:
    clean = _CLEAN_RE.sub("", plate_text.upper())
    for pattern, country in _NATIONALITY_RULES:
        if pattern.match(clean):
            return country
    return "Unknown"
