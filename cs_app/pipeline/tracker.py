import logging
from dataclasses import dataclass, field

from cs_app.config import TRACK_GRACE_FRAMES, STATIONARY_START_FRAMES, ANPR_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

# Frames to keep a departed track in the re-ID pool
_REID_WINDOW_FRAMES = 90
# Minimum IoU to consider a new track the same vehicle as a departed one
_REID_IOU_THRESHOLD = 0.25


@dataclass
class TrackState:
    observation_id: str
    first_seen_frame: int
    last_seen_frame: int
    plate_confirmed: bool = False      # True once confidence >= ANPR_CONFIDENCE_THRESHOLD
    plate_locked: bool = False         # True once plate is permanently locked (no more updates)
    best_plate_confidence: float = 0.0 # Highest confidence seen so far for this vehicle
    is_stationary_at_start: bool = False
    bbox_last: list[float] = field(default_factory=list)


@dataclass
class _DepartedEntry:
    observation_id: str
    bbox: list[float]
    departed_frame: int
    plate_confirmed: bool
    plate_locked: bool
    best_plate_confidence: float


class TrackStateManager:
    """
    Detects newly appeared and departed track IDs between frames.

    Grace period: a track must be absent for > TRACK_GRACE_FRAMES consecutive
    frames before it is declared departed (handles brief detection gaps).

    Re-identification: when a new track ID appears, its bbox is compared via IoU
    against recently departed tracks. If a match is found the new ID is silently
    mapped to the existing observation — avoiding duplicate observation records
    for the same physical vehicle.
    """

    def __init__(self):
        self._states: dict[int, TrackState] = {}
        self._absent_counter: dict[int, int] = {}
        # Pool of recently-departed tracks available for re-ID
        self._departed_pool: list[_DepartedEntry] = []

    def update(
        self,
        current_detections: list,
        current_frame_idx: int,
    ) -> tuple[list[int], list[int]]:
        """
        Returns:
            new_track_ids:      IDs seen for the first time (no re-ID match found)
            departed_track_ids: IDs absent > TRACK_GRACE_FRAMES
        """
        current_ids = {d.track_id for d in current_detections}
        known_ids   = set(self._states.keys())

        # Update last_seen and bbox for visible tracks
        for det in current_detections:
            tid = det.track_id
            if tid in self._states:
                self._states[tid].last_seen_frame = current_frame_idx
                self._states[tid].bbox_last = det.bbox_xyxy
            if tid in self._absent_counter:
                del self._absent_counter[tid]

        # Increment absent counter; collect truly departed
        departed_ids = []
        for tid in known_ids - current_ids:
            self._absent_counter[tid] = self._absent_counter.get(tid, 0) + 1
            if self._absent_counter[tid] > TRACK_GRACE_FRAMES:
                departed_ids.append(tid)

        for tid in departed_ids:
            state = self._states.pop(tid)
            del self._absent_counter[tid]
            self._departed_pool.append(_DepartedEntry(
                observation_id=state.observation_id,
                bbox=state.bbox_last,
                departed_frame=current_frame_idx,
                plate_confirmed=state.plate_confirmed,
                plate_locked=state.plate_locked,
                best_plate_confidence=state.best_plate_confidence,
            ))
            logger.debug("Track %d departed (obs=%s)", tid, state.observation_id)

        # Expire old re-ID pool entries
        self._departed_pool = [
            e for e in self._departed_pool
            if current_frame_idx - e.departed_frame <= _REID_WINDOW_FRAMES
        ]

        # Classify new track IDs: genuine new vehicle or re-identified?
        raw_new_ids = current_ids - known_ids
        new_ids = []
        det_by_id = {d.track_id: d for d in current_detections}

        for tid in raw_new_ids:
            det = det_by_id[tid]
            match = _find_reid_match(det.bbox_xyxy, self._departed_pool)
            if match is not None:
                # Re-use the existing observation — register under the new track ID
                self._departed_pool.remove(match)
                self._states[tid] = TrackState(
                    observation_id=match.observation_id,
                    first_seen_frame=current_frame_idx,
                    last_seen_frame=current_frame_idx,
                    plate_confirmed=match.plate_confirmed,
                    plate_locked=match.plate_locked,
                    best_plate_confidence=match.best_plate_confidence,
                    bbox_last=det.bbox_xyxy,
                )
                logger.debug(
                    "Re-ID: new track %d → existing obs %s (IoU match)",
                    tid, match.observation_id,
                )
            else:
                new_ids.append(tid)

        return new_ids, departed_ids

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

    def update_plate_confidence(self, track_id: int, confidence: float):
        """Record a new plate confidence reading. Locks permanently at threshold."""
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
    """Compute Intersection-over-Union of two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _find_reid_match(
    bbox: list[float],
    pool: list[_DepartedEntry],
) -> _DepartedEntry | None:
    """Return the best IoU match from the departed pool, or None."""
    best, best_iou = None, _REID_IOU_THRESHOLD
    for entry in pool:
        iou = _iou(bbox, entry.bbox)
        if iou > best_iou:
            best_iou = iou
            best = entry
    return best
