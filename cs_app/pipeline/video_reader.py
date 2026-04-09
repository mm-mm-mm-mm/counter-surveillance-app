import logging
from datetime import datetime, timezone
from pathlib import Path

import av
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoReader:
    """Handles video metadata extraction (via PyAV) and frame iteration (via OpenCV)."""

    def __init__(self, path: Path):
        self.path = path

    def get_creation_time(self) -> datetime:
        """
        Return the video's creation datetime as UTC.
        Fallback chain:
          1. PyAV container metadata 'creation_time'
          2. QuickTime 'com.apple.quicktime.creationdate' tag
          3. File system modification time
        """
        try:
            container = av.open(str(self.path))
            meta = container.metadata

            # Standard MP4/MOV tag
            ct = meta.get("creation_time")
            if ct:
                container.close()
                return _parse_av_datetime(ct)

            # QuickTime-specific tag
            qt = meta.get("com.apple.quicktime.creationdate")
            if qt:
                container.close()
                return _parse_av_datetime(qt)

            container.close()
        except Exception as e:
            logger.warning("PyAV metadata read failed: %s", e)

        # Fallback: file modification time (treated as UTC)
        logger.warning(
            "No creation_time found in '%s'. Using file modification date.", self.path.name
        )
        mtime = self.path.stat().st_mtime
        return datetime.fromtimestamp(mtime, tz=timezone.utc)

    def get_duration(self) -> float:
        """Return total video duration in seconds."""
        cap = cv2.VideoCapture(str(self.path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return frame_count / fps if fps > 0 else 0.0

    def get_fps(self) -> float:
        cap = cv2.VideoCapture(str(self.path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        return fps

    def iter_frames(self, skip: int = 1):
        """
        Yield (frame_index, elapsed_seconds, bgr_ndarray) for each processed frame.
        skip=2 processes every other frame, etc.
        """
        cap = cv2.VideoCapture(str(self.path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % skip == 0:
                elapsed = frame_idx / fps
                yield frame_idx, elapsed, frame

            frame_idx += 1

        cap.release()

    def has_creation_time_metadata(self) -> bool:
        """Returns True if the file has an embedded creation_time tag."""
        try:
            container = av.open(str(self.path))
            meta = container.metadata
            has = bool(meta.get("creation_time") or meta.get("com.apple.quicktime.creationdate"))
            container.close()
            return has
        except Exception:
            return False


def _parse_av_datetime(value: str) -> datetime:
    """Parse ISO 8601 datetime strings as returned by PyAV metadata."""
    value = value.strip()
    # PyAV returns strings like "2024-03-15T14:32:07.000000Z"
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
    ):
        try:
            dt = datetime.strptime(value, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    # Last resort: return epoch so the app keeps running
    logger.warning("Could not parse datetime string: %r — using epoch", value)
    return datetime(1970, 1, 1, tzinfo=timezone.utc)
