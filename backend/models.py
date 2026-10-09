"""
The only thing the kiosk stores: sightings the B.O.T backend could not take yet.

Faces, employees and attendance all live in the backend now (FaceProfile,
PunchLog, Attendance). This outbox exists for one case: the backend was
unreachable when a face was recognised. The sighting is kept here with its
real time and sent when the backend answers again; the backend's unique tap
index makes a re-sent sighting count once.
"""
from sqlalchemy import Column, Integer, String, Float, Text
from database import Base


class OutboxTap(Base):
    __tablename__ = "outbox_taps"
    event_id = Column(String, primary_key=True)
    kiosk_key = Column(Text, nullable=False)     # the kiosk key it must be sent with
    payload = Column(Text, nullable=False)       # JSON: employeeId, tapTime, score
    attempts = Column(Integer, default=0)
    next_try = Column(Float, default=0.0)        # unix time
    created_at = Column(Float, default=0.0)
