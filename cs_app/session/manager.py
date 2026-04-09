import asyncio
import base64
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2

from cs_app.config import FRAME_SKIP, TARGET_FPS, ANPR_CONFIDENCE_THRESHOLD
from cs_app.db.crud import (
    create_observation,
    update_observation_plate,
    update_observation_last_seen,
    get_all_observations,
)
from cs_app.db.session import AsyncSessionLocal, drop_and_recreate_db
from cs_app.pipeline.anpr import ANPRProcessor
from cs_app.pipeline.color_classifier import classify_vehicle_color
from cs_app.pipeline.detector import VehicleDetector
from cs_app.pipeline.snapshot import save_snapshot
from cs_app.pipeline.tracker import TrackStateManager
from cs_app.pipeline.video_reader import VideoReader
from cs_app.session.exporter import export_csv

logger = logging.getLogger(__name__)
DEBUG_TIMING = os.environ.get("DEBUG_TIMING", "").lower() in ("1", "true", "yes")


class ProcessingSession:
    def __init__(self, video_path: Path):
        self.video_path = video_path
        self._obs_counter = 0

    def _next_obs_id(self) -> str:
        self._obs_counter += 1
        return f"{self.video_path.stem}_{self._obs_counter}"

    async def run(self, websocket):
        reader = VideoReader(self.video_path)

        # Warn if no embedded creation time
        if not reader.has_creation_time_metadata():
            await websocket.send_text(json.dumps({
                "type": "metadata_warning",
                "message": f"No creation time found in '{self.video_path.name}'. Using file modification date."
            }))

        t0 = reader.get_creation_time()
        duration = reader.get_duration()
        tracker = TrackStateManager()

        # Fresh DB for this session
        await drop_and_recreate_db()

        # Load models from app.state if pre-warmed, else instantiate here
        try:
            from cs_app.main import app
            detector: VehicleDetector = app.state.detector
            anpr: ANPRProcessor = app.state.anpr
        except Exception:
            detector = VehicleDetector()
            anpr = ANPRProcessor()

        min_send_interval = 1.0 / TARGET_FPS
        last_send_time = 0.0

        async with AsyncSessionLocal() as db:
            for frame_idx, elapsed, frame_bgr in reader.iter_frames(skip=FRAME_SKIP):
                t_frame_start = time.monotonic()

                current_time: datetime = t0 + timedelta(seconds=elapsed)

                # --- Detection + Tracking ---
                t_det = time.monotonic()
                detections = detector.detect_and_track(frame_bgr)
                det_ms = (time.monotonic() - t_det) * 1000

                new_ids, departed_ids = tracker.update(detections, frame_idx)

                # --- Handle departed vehicles ---
                for tid in departed_ids:
                    state = tracker.get_state(tid)
                    if state:
                        await update_observation_last_seen(db, state.observation_id, current_time)

                # Build a lookup for this frame's detections
                det_by_id = {d.track_id: d for d in detections}

                # --- Handle new vehicles ---
                for tid in new_ids:
                    det = det_by_id.get(tid)
                    if det is None:
                        continue

                    obs_id = self._next_obs_id()
                    make, model = detector.get_make_model(det.class_name)
                    color = classify_vehicle_color(frame_bgr, det.bbox_xyxy)

                    # Stationary vehicles at video start use t0
                    first_seen = t0 if frame_idx <= 30 else current_time

                    tracker.register(tid, obs_id, frame_idx, det.bbox_xyxy)
                    await create_observation(db, obs_id, tid, first_seen, make, model, color)
                    save_snapshot(frame_bgr, det.bbox_xyxy, obs_id)

                    # Run ANPR on new vehicle
                    t_anpr = time.monotonic()
                    anpr_result = anpr.read_plate(frame_bgr, det.bbox_xyxy)
                    anpr_ms = (time.monotonic() - t_anpr) * 1000

                    if anpr_result.text:
                        await update_observation_plate(
                            db, obs_id,
                            anpr_result.text,
                            anpr_result.plate_color,
                            anpr_result.category,
                            anpr_result.nationality,
                        )
                        if not anpr.should_retry(anpr_result):
                            tracker.mark_plate_confirmed(tid)
                    else:
                        anpr_ms = 0.0

                # --- Retry ANPR for unconfirmed plates ---
                for tid in tracker.all_active_track_ids():
                    state = tracker.get_state(tid)
                    if state and not state.plate_confirmed and tid not in new_ids:
                        det = det_by_id.get(tid)
                        if det:
                            t_anpr = time.monotonic()
                            anpr_result = anpr.read_plate(frame_bgr, det.bbox_xyxy)
                            anpr_ms = (time.monotonic() - t_anpr) * 1000
                            if anpr_result.text:
                                await update_observation_plate(
                                    db, state.observation_id,
                                    anpr_result.text,
                                    anpr_result.plate_color,
                                    anpr_result.category,
                                    anpr_result.nationality,
                                )
                                if not anpr.should_retry(anpr_result):
                                    tracker.mark_plate_confirmed(tid)

                # --- Build and send frame message ---
                now = time.monotonic()
                if now - last_send_time >= min_send_interval:
                    t_enc = time.monotonic()
                    _, buf = cv2.imencode(
                        ".jpg", frame_bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, 75]
                    )
                    frame_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                    enc_ms = (time.monotonic() - t_enc) * 1000

                    active_obs = await _build_active_obs(db, tracker, det_by_id, t0, elapsed)

                    msg = {
                        "type": "frame",
                        "frame_data": frame_b64,
                        "timestamp_display": current_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                        "elapsed_seconds": round(elapsed, 2),
                        "active_observations": active_obs,
                        "new_observations": [
                            tracker.get_state(tid).observation_id
                            for tid in new_ids
                            if tracker.get_state(tid)
                        ],
                        "departed_observations": [],  # populated via departed_ids above
                    }
                    await websocket.send_text(json.dumps(msg))
                    last_send_time = now

                    if DEBUG_TIMING:
                        total_ms = (time.monotonic() - t_frame_start) * 1000
                        logger.debug(
                            "frame=%d det=%.1fms enc=%.1fms total=%.1fms",
                            frame_idx, det_ms, enc_ms, total_ms
                        )

                # Yield to event loop
                await asyncio.sleep(0)

            # --- Session end ---
            await self._finalize(db, tracker, t0, duration, websocket)

    async def _finalize(self, db, tracker, t0, duration, websocket):
        end_time = t0 + timedelta(seconds=duration)

        for tid in tracker.all_active_track_ids():
            state = tracker.get_state(tid)
            if state:
                await update_observation_last_seen(db, state.observation_id, end_time)

        all_obs = await get_all_observations(db)
        csv_path = await export_csv(all_obs, self.video_path.name)

        final_list = [_obs_to_dict(o) for o in all_obs]

        await websocket.send_text(json.dumps({
            "type": "session_end",
            "csv_path": str(csv_path.relative_to(csv_path.parent.parent)),
            "final_observations": final_list,
        }))
        logger.info("Session complete. CSV: %s", csv_path)


async def _build_active_obs(db, tracker, det_by_id, t0, elapsed_now):
    """Build the list of active observation dicts for the current frame message."""
    from cs_app.db.crud import get_all_observations
    all_obs = await get_all_observations(db)
    obs_map = {o.internal_track_id: o for o in all_obs}

    active = []
    for tid in tracker.all_active_track_ids():
        obs = obs_map.get(tid)
        state = tracker.get_state(tid)
        det = det_by_id.get(tid)
        if obs and state and det:
            first_seen_epoch = obs.date_time_first_observation.timestamp() if obs.date_time_first_observation else 0
            current_epoch = (t0 + timedelta(seconds=elapsed_now)).timestamp()
            elapsed_since_first = max(0.0, current_epoch - first_seen_epoch)

            active.append({
                "observation_id": obs.observation_id,
                "bbox": [int(v) for v in det.bbox_xyxy],
                "plate_text": obs.vehicle_licence_plate or "",
                "plate_color": obs.vehicle_licence_plate_color or "white",
                "category": obs.category or "normal",
                "make": obs.vehicle_make or "",
                "model": obs.vehicle_model or "",
                "color": obs.vehicle_color or "",
                "nationality": obs.vehicle_licence_plate_nationality or "",
                "first_seen": obs.date_time_first_observation.strftime("%Y-%m-%d %H:%M:%S UTC") if obs.date_time_first_observation else "",
                "elapsed_since_first": round(elapsed_since_first, 1),
            })
    return active


def _obs_to_dict(obs) -> dict:
    return {
        "observation_id": obs.observation_id,
        "plate_text": obs.vehicle_licence_plate or "",
        "plate_color": obs.vehicle_licence_plate_color or "white",
        "category": obs.category or "normal",
        "make": obs.vehicle_make or "",
        "model": obs.vehicle_model or "",
        "color": obs.vehicle_color or "",
        "nationality": obs.vehicle_licence_plate_nationality or "",
        "first_seen": obs.date_time_first_observation.strftime("%Y-%m-%d %H:%M:%S UTC") if obs.date_time_first_observation else "",
        "last_seen": obs.date_time_last_observation.strftime("%Y-%m-%d %H:%M:%S UTC") if obs.date_time_last_observation else "",
    }
