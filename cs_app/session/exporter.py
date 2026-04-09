import csv
import logging
from pathlib import Path

from cs_app.config import SESSION_DATA_DIR
from cs_app.db.models import Observation

logger = logging.getLogger(__name__)

_EXPORT_FIELDS = [
    "observation_id",
    "date_time_first_observation",
    "date_time_last_observation",
    "vehicle_make",
    "vehicle_model",
    "vehicle_color",
    "vehicle_licence_plate",
    "vehicle_licence_plate_color",
    "vehicle_licence_plate_nationality",
]


async def export_csv(observations: list[Observation], video_filename: str) -> Path:
    stem = Path(video_filename).stem
    out_path = SESSION_DATA_DIR / f"{stem}.csv"

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_EXPORT_FIELDS)
        writer.writeheader()
        for obs in observations:
            writer.writerow({
                "observation_id": obs.observation_id,
                "date_time_first_observation": _fmt(obs.date_time_first_observation),
                "date_time_last_observation": _fmt(obs.date_time_last_observation),
                "vehicle_make": obs.vehicle_make or "",
                "vehicle_model": obs.vehicle_model or "",
                "vehicle_color": obs.vehicle_color or "",
                "vehicle_licence_plate": obs.vehicle_licence_plate or "",
                "vehicle_licence_plate_color": obs.vehicle_licence_plate_color or "",
                "vehicle_licence_plate_nationality": obs.vehicle_licence_plate_nationality or "",
            })

    logger.info("CSV exported to %s", out_path)
    return out_path


def _fmt(dt) -> str:
    if dt is None:
        return ""
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
