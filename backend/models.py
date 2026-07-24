from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, Float
from sqlalchemy.orm import relationship
from database import Base
import datetime

class Employee(Base):
    __tablename__ = "employees"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    phone = Column(String, default="")
    photo_path = Column(String)
    face_encoding = Column(String)
    is_active = Column(Boolean, default=True)
    
    # Financial Details
    monthly_salary = Column(Float, default=0.0)
    hourly_rate = Column(Float, default=0.0)
    bank_name = Column(String, default="")
    bank_account = Column(String, default="")
    bank_ifsc = Column(String, default="")

    # Links this employee to the Node/HRMS backend's tenant + user record, so
    # camera-detected punches can be synced there. Blank means "not linked" —
    # sync is skipped for that employee.
    external_admin_id = Column(String, default="")
    external_user_id = Column(String, default="")

class AttendanceLog(Base):
    __tablename__ = "attendance_logs"
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"))
    action = Column(String)  # 'PUNCH_IN', 'PUNCH_OUT', 'ACTIVITY_UPDATE'
    reason = Column(String)  # 'FACE_MATCH', 'LEFT_DESK', 'PHONE_DETECTED', 'LAPTOP_DETECTED'
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    employee = relationship("Employee")

class ActivityStats(Base):
    __tablename__ = "activity_stats"
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"))
    date = Column(String)  # YYYY-MM-DD
    work_seconds = Column(Integer, default=0)
    phone_seconds = Column(Integer, default=0)
    
    employee = relationship("Employee")
