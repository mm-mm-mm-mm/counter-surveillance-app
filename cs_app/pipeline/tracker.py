import logging
from dataclasses import dataclass, field

from cs_app.config import (
    TRACK_GRACE_FRAMES, STATIONARY_START_FRAMES, ANPR_CONFIDENCE_THRESHOLD,
    REID_WINDOW_FRAMES, REID_IOU_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Centroid distance threshold as fraction of frame diagonal — fallback when IoU is low
_REID_CENTROID_FRACTION = 0.12


@dataclass
class TrackState:
    observation_id: str
    first_seen_frame: int
    last_seen_frame: int
    plate_confirmed: bool = False
    plate_locked: bool = False
    best_plate_confidence: float = 0.0
    last_anpr_frame: int = -1
    is_stationary_at_start: bool = False
    bbox_last: list[float] = field(default_factory=list)


@dataclass
class _ReidCandidate:
    """Unified re-ID candidate — covers both grace-period and departed tracks."""
    observation_id: str
    bbox: list[float]
    plate_confirmed: bool
    plate_locked: bool
    best_plate_confidence: float
    last_anpr_frame: int
    # For grace-period tracks we also hold the old track ID so we can clean it up
    grace_track_id: int | None = None
    # For departed-pool entries we hold a reference so we can remove it
    departed_entry: object = None


class TrackStateManager:
    """
    Tracks vehicle identities across frames.

    Two-tier re-identification:
    1. Grace period (absent but not yet departed): when a new track ID appears
       its bbox is checked against ALL currently absent tracks. This handles the
       common case where ByteTrack reassigns a new ID immediately after losing
       the old one — before the grace period expires.
    2. Departed pool: tracks that have fully expired are kept for REID_WINDOW_FRAMES
       additional frames to catch longer occlusions.

    A match is accepted if IoU ≥ REID_IOU_THRESHOLD OR the bounding-box
    centroids are within _REID_CENTROID_FRACTION of the frame diagonal —
    whichever fires first.
    """

    def __init__(self):
        self._states: dict[int, TrackState] = {}
        self._absent_counter: dict[int, int] = {}
        self._departed_pool: list[_ReidCandidate] = []
        self._frame_diag: float = 0.0  # updated lazily from bbox sizes

    def update(
        self,
        current_detections: list,
        current_frame_idx: int,
    ) -> tuple[list[int], list[int]]:
        """
        Returns:
            new_track_ids:      genuinely new vehicles (no re-ID match)
            departed_track_ids: vehicles that have been absent > TRACK_GRACE_FRAMES
        """
        current_ids = {d.track_id for d in current_detections}
        known_ids   = set(self._states.keys())
        det_by_id   = {d.track_id: d for d in current_detections}

        # Update frame diagonal estimate from bbox sizes (used for centroid threshold)
        for det in current_detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            diag = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            if diag > self._frame_diag:
                self._frame_diag = min(diag * 8, 3000.0)

        # Update last_seen and bbox for currently visible tracks
        for det in current_detections:
            tid = det.track_id
            if tid in self._states:
                self._states[tid].last_seen_frame = current_frame_idx
                self._states[tid].bbox_last = det.bbox_xyxy
            # If a track reappeared during its grace period, reset counter
            if tid in self._absent_counter:
                del self._absent_counter[tid]

        # Count absent frames; move to departed pool when grace expires
        departed_ids = []
        for tid in known_ids - current_ids:
            self._absent_counter[tid] = self._absent_counter.get(tid, 0) + 1
            if self._absent_counter[tid] > TRACK_GRACE_FRAMES:
                departed_ids.append(tid)

        for tid in departed_ids:
            state = self._states.pop(tid)
            del self._absent_counter[tid]
            self._departed_pool.append(_ReidCandidate(
                observation_id=state.observation_id,
                bbox=state.bbox_last,
                plate_confirmed=state.plate_confirmed,
                plate_locked=state.plate_locked,
                best_plate_confidence=state.best_plate_confidence,
                last_anpr_frame=state.last_anpr_frame,
            ))
            logger.debug("Track %d fully departed (obs=%s)", tid, state.observation_id)

        # Expire old departed-pool entries
        self._departed_pool = [
            e for e in self._departed_pool
            if current_frame_idx - (getattr(e, '_departed_frame', current_frame_idx)) <= REID_WINDOW_FRAMES
        ]

        # Build the full re-ID candidate pool:
        #   tier-1: grace-period tracks (still in _states but absent)
        #   tier-2: fully departed pool
        candidates: list[_ReidCandidate] = []

        for tid, cnt in self._absent_counter.items():
            state = self._states.get(tid)
            if state and state.bbox_last:
                candidates.append(_ReidCandidate(
                    observation_id=state.observation_id,
                    bbox=state.bbox_last,
                    plate_confirmed=state.plate_confirmed,
                    plate_locked=state.plate_locked,
                    best_plate_confidence=state.best_plate_confidence,
                    last_anpr_frame=state.last_anpr_frame,
                    grace_track_id=tid,
                ))

        candidates.extend(self._departed_pool)

        # Classify each new track ID
        raw_new_ids = current_ids - known_ids
        new_ids = []

        for tid in raw_new_ids:
            det = det_by_id[tid]
            match = self._find_match(det.bbox_xyxy, candidates)

            if match is not None:
                # Absorb the matched candidate into the new track ID
                if match.grace_track_id is not None:
                    # Remove the old grace-period track
                    old_tid = match.grace_track_id
                    self._states.pop(old_tid, None)
                    self._absent_counter.pop(old_tid, None)
                    candidates = [c for c in candidates if c.grace_track_id != old_tid]
                elif match.departed_entry is not None:
                    self._departed_pool.remove(match.departed_entry)
                else:
                    # Entry came directly from departed pool list
                    if match in self._departed_pool:
                        self._departed_pool.remove(match)

                self._states[tid] = TrackState(
                    observation_id=match.observation_id,
                    first_seen_frame=current_frame_idx,
                    last_seen_frame=current_frame_idx,
                    plate_confirmed=match.plate_confirmed,
                    plate_locked=match.plate_locked,
                    best_plate_confidence=match.best_plate_confidence,
                    last_anpr_frame=match.last_anpr_frame,
                    bbox_last=det.bbox_xyxy,
                )
                # Remove from candidates so it can't match a second new track
                candidates = [c for c in candidates if c.observation_id != match.observation_id]
                logger.debug("Re-ID: track %d → obs %s", tid, match.observation_id)
            else:
                new_ids.append(tid)

        return new_ids, departed_ids

    def _find_match(
        self, bbox: list[float], candidates: list[_ReidCandidate]
    ) -> _ReidCandidate | None:
        best_score = -1.0
        best = None
        threshold = _REID_CENTROID_FRACTION * self._frame_diag if self._frame_diag > 0 else 999

        for c in candidates:
            if not c.bbox:
                continue
            iou = _iou(bbox, c.bbox)
            dist = _centroid_dist(bbox, c.bbox)

            # Score: IoU takes priority; use inverted distance as tiebreaker
            if iou >= REID_IOU_THRESHOLD:
                score = 1.0 + iou
            elif dist <= threshold:
                score = 1.0 - (dist / (threshold + 1e-6))
            else:
                continue

            if score > best_score:
                best_score = score
                best = c

        return best

    def register(self, track_id: int, observation_id: str, frame_idx: int, bbox: list[float]):
        is_stationary = frame_idx <= STATIONARY_START_FRAMES
        self._states[track_id] = TrackState(
            observation_id=observation_id,
            first_seen_frame=frame_idx,
            last_seen_frame=frame_idx,
            is_stationary_at_start=is_stationary,
            bbox_last=bbox,
        )

    def get_state(self, track_id: int) -> TrackState | None:
        return self._states.get(track_id)

    def all_active_track_ids(self) -> list[int]:
        return list(self._states.keys())

    def record_anpr_attempt(self, track_id: int, frame_idx: int):
        if track_id in self._states:
            self._states[track_id].last_anpr_frame = frame_idx

    def anpr_due(self, track_id: int, current_frame: int, retry_interval: int) -> bool:
        state = self._states.get(track_id)
        if not state:
            return False
        if state.last_anpr_frame < 0:
            return True
        return (current_frame - state.last_anpr_frame) >= retry_interval

    def update_plate_confidence(self, track_id: int, confidence: float):
        state = self._states.get(track_id)
        if state and not state.plate_locked:
            if confidence > state.best_plate_confidence:
                state.best_plate_confidence = confidence
            state.plate_confirmed = True
            if confidence >= ANPR_CONFIDENCE_THRESHOLD:
                state.plate_locked = True

    def is_plate_locked(self, track_id: int) -> bool:
        state = self._states.get(track_id)
        return state.plate_locked if state else False

    def mark_plate_confirmed(self, track_id: int):
        if track_id in self._states:
            self._states[track_id].plate_confirmed = True


def _iou(a: list[float], b: list[float]) -> float:
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _centroid_dist(a: list[float], b: list[float]) -> float:
    ax = (a[0] + a[2]) / 2;  ay = (a[1] + a[3]) / 2
    bx = (b[0] + b[2]) / 2;  by = (b[1] + b[3]) / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
