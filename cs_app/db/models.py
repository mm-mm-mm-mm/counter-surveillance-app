from sqlalchemy import Column, Integer, String, DateTime, Float
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Observation(Base):
    __tablename__ = "observations"

    observation_id = Column(String, primary_key=True)
    date_time_first_observation = Column(DateTime, nullable=True)
    date_time_last_observation = Column(DateTime, nullable=True)
    vehicle_make = Column(String, nullable=True)
    vehicle_model = Column(String, nullable=True)
    vehicle_color = Column(String, nullable=True)
    vehicle_licence_plate = Column(String, nullable=True)
    vehicle_licence_plate_color = Column(String, nullable=True)
    vehicle_licence_plate_nationality = Column(String, nullable=True)
    category = Column(String, nullable=True)

    # Internal field — not exported to CSV
    internal_track_id = Column(Integer, nullable=True, index=True)
