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
# Checked in order; first match wins
_NATIONALITY_RULES: list[tuple[re.Pattern, str]] = [
    # UK: AB12 CDE or AB12CDE
    (re.compile(r"^[A-Z]{2}\d{2}[A-Z]{3}$"), "GB"),
    # Sweden: ABC 123 or ABC123
    (re.compile(r"^[A-Z]{3}\d{3}$"), "SE"),
    # Norway: AB 12345
    (re.compile(r"^[A-Z]{2}\d{5}$"), "NO"),
    # Denmark: AB 12 345
    (re.compile(r"^[A-Z]{2}\d{2}\d{3}$"), "DK"),
    # Germany: ABC DE 1234 (variable length)
    (re.compile(r"^[A-Z]{1,3}[A-Z]{1,2}\d{1,4}[HE]?$"), "DE"),
    # France: AB-123-CD
    (re.compile(r"^[A-Z]{2}\d{3}[A-Z]{2}$"), "FR"),
    # Netherlands: AB-123-C or similar
    (re.compile(r"^[A-Z]{2}\d{3}[A-Z]$"), "NL"),
    # Poland: AB 12345 or AB1 C234
    (re.compile(r"^[A-Z]{2,3}\d{4,5}$"), "PL"),
    # Finland: ABC-123
    (re.compile(r"^[A-Z]{2,3}\d{3}$"), "FI"),
    # Generic 2-letter + digits fallback
    (re.compile(r"^[A-Z]{1,3}\d{3,4}[A-Z]{0,3}$"), "Unknown"),
]

# Remove spaces, hyphens, dots for matching
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
        Attempt to read the licence plate from a vehicle bounding box.
        Returns an ANPRResult; text may be empty if nothing detected.
        """
        plate_crop = extract_plate_region(frame_bgr, bbox_xyxy)

        plate_color, category = classify_plate_color(plate_crop)

        if plate_crop.size == 0:
            return ANPRResult("", 0.0, plate_color, category, "Unknown")

        try:
            ocr_results = self.reader.readtext(plate_crop, detail=1)
        except Exception as e:
            logger.warning("EasyOCR failed: %s", e)
            return ANPRResult("", 0.0, plate_color, category, "Unknown")

        best_text = ""
        best_conf = 0.0

        for (_bbox, text, conf) in ocr_results:
            clean = _CLEAN_RE.sub("", text.upper())
            # Filter: must be alphanumeric, 4–10 chars
            if not re.match(r"^[A-Z0-9]{4,10}$", clean):
                continue
            if conf > best_conf:
                best_conf = conf
                best_text = clean

        nationality = infer_nationality(best_text) if best_text else "Unknown"

        return ANPRResult(
            text=best_text,
            confidence=best_conf,
            plate_color=plate_color,
            category=category,
            nationality=nationality,
        )

    def should_retry(self, result: ANPRResult) -> bool:
        """True if the plate was not confidently read and should be retried."""
        return not result.text or result.confidence < ANPR_CONFIDENCE_THRESHOLD


def infer_nationality(plate_text: str) -> str:
    clean = _CLEAN_RE.sub("", plate_text.upper())
    for pattern, country in _NATIONALITY_RULES:
        if pattern.match(clean):
            return country
    return "Unknown"
