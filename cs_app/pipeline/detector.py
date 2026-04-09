import logging
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO
from cs_app.config import DEVICE

logger = logging.getLogger(__name__)

# COCO class IDs for motor vehicles
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
COCO_CLASS_NAMES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}


@dataclass
class DetectionResult:
    track_id: int
    bbox_xyxy: list[float]  # [x1, y1, x2, y2]
    class_name: str
    confidence: float


class VehicleDetector:
    def __init__(self):
        logger.info("Loading YOLOv8l model on device=%s ...", DEVICE)
        self.model = YOLO("yolov8l.pt")
        self.model.to(DEVICE)
        logger.info("YOLOv8l loaded.")

    def detect_and_track(self, frame_bgr: np.ndarray) -> list[DetectionResult]:
        """Run YOLOv8 detection + ByteTrack on a single BGR frame."""
        results = self.model.track(
            frame_bgr,
            persist=True,
            tracker="bytetrack.yaml",
            classes=VEHICLE_CLASSES,
            verbose=False,
            device=DEVICE,
        )

        detections = []
        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        if boxes.id is None:
            # Tracker hasn't assigned IDs yet (first frame or lost all tracks)
            return detections

        for box, track_id, cls, conf in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.id.cpu().numpy().astype(int),
            boxes.cls.cpu().numpy().astype(int),
            boxes.conf.cpu().numpy(),
        ):
            detections.append(DetectionResult(
                track_id=int(track_id),
                bbox_xyxy=box.tolist(),
                class_name=COCO_CLASS_NAMES.get(cls, "Vehicle"),
                confidence=float(conf),
            ))

        return detections

    def reset_tracker(self):
        """Clear ByteTrack state. Call at the start of each new session."""
        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            self.model.predictor = None
        logger.debug("Tracker state reset.")

    def get_make_model(self, class_name: str) -> tuple[str, str]:
        """
        V1: returns coarse class name as make, model unknown.
        Replace this method with a fine-grained classifier in a future version.
        """
        return class_name, "Unknown"
