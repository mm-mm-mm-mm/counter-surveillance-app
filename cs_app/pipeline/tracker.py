import logging
from dataclasses import dataclass, field

from cs_app.config import TRACK_GRACE_FRAMES, STATIONARY_START_FRAMES

logger = logging.getLogger(__name__)


@dataclass
class TrackState:
    observation_id: str
    first_seen_frame: int
    last_seen_frame: int
    plate_confirmed: bool = False
    is_stationary_at_start: bool = False
    bbox_last: list[float] = field(default_factory=list)


class TrackStateManager:
    """
    Detects newly appeared and departed track IDs between frames.
    Applies a grace period before declaring a track departed (handles brief detection gaps).
    """

    def __init__(self):
        self._states: dict[int, TrackState] = {}
        # Maps track_id → frames-since-last-seen (counts up during grace period)
        self._absent_counter: dict[int, int] = {}

    def update(
        self,
        current_detections: list,  # list[DetectionResult]
        current_frame_idx: int,
    ) -> tuple[list[int], list[int]]:
        """
        Process detections for the current frame.

        Returns:
            new_track_ids: track IDs seen for the first time
            departed_track_ids: track IDs that have been absent > TRACK_GRACE_FRAMES
        """
        current_ids = {d.track_id for d in current_detections}
        known_ids = set(self._states.keys())

        # Update last_seen and bbox for all currently visible tracks
        for det in current_detections:
            tid = det.track_id
            if tid in self._states:
                self._states[tid].last_seen_frame = current_frame_idx
                self._states[tid].bbox_last = det.bbox_xyxy
            # Reset absent counter if track reappeared
            if tid in self._absent_counter:
                del self._absent_counter[tid]

        # Find newly appeared track IDs
        new_ids = current_ids - known_ids

        # Increment absent counter for tracks not seen this frame
        absent_ids = known_ids - current_ids
        departed_ids = []
        for tid in absent_ids:
            self._absent_counter[tid] = self._absent_counter.get(tid, 0) + 1
            if self._absent_counter[tid] > TRACK_GRACE_FRAMES:
                departed_ids.append(tid)

        # Remove departed tracks from state
        for tid in departed_ids:
            del self._states[tid]
            del self._absent_counter[tid]
            logger.debug("Track %d departed (grace exceeded)", tid)

        return list(new_ids), departed_ids

    def register(
        self,
        track_id: int,
        observation_id: str,
        frame_idx: int,
        bbox: list[float],
    ):
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

    def mark_plate_confirmed(self, track_id: int):
        if track_id in self._states:
            self._states[track_id].plate_confirmed = True
