from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from cs_app.db.models import Observation


async def create_observation(
    db: AsyncSession,
    observation_id: str,
    internal_track_id: int,
    date_time_first: datetime,
    make: str,
    model: str,
    color: str,
) -> Observation:
    obs = Observation(
        observation_id=observation_id,
        internal_track_id=internal_track_id,
        date_time_first_observation=date_time_first,
        vehicle_make=make,
        vehicle_model=model,
        vehicle_color=color,
    )
    db.add(obs)
    await db.commit()
    return obs


async def update_observation_plate(
    db: AsyncSession,
    observation_id: str,
    plate_text: str,
    plate_color: str,
    nationality: str,
) -> None:
    result = await db.execute(
        select(Observation).where(Observation.observation_id == observation_id)
    )
    obs = result.scalar_one_or_none()
    if obs:
        obs.vehicle_licence_plate = plate_text
        obs.vehicle_licence_plate_color = plate_color
        obs.vehicle_licence_plate_nationality = nationality
        await db.commit()


async def update_observation_last_seen(
    db: AsyncSession,
    observation_id: str,
    date_time_last: datetime,
) -> None:
    result = await db.execute(
        select(Observation).where(Observation.observation_id == observation_id)
    )
    obs = result.scalar_one_or_none()
    if obs:
        obs.date_time_last_observation = date_time_last
        await db.commit()


async def get_all_observations(db: AsyncSession) -> list[Observation]:
    result = await db.execute(select(Observation))
    return list(result.scalars().all())
