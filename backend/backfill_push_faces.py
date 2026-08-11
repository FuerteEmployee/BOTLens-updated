"""One-time backfill: push every already-registered employee's face up to
Node. push_face_to_node() only fires going forward, on add/edit — employees
registered before this feature existed (i.e. everyone except whoever's been
manually re-saved since) were never pushed. Run once, directly against the
instance that actually has these employees registered locally (production),
after this branch is deployed there.

Usage: python backfill_push_faces.py
"""
from database import SessionLocal
from models import Employee
from engine import push_face_to_node

db = SessionLocal()
try:
    employees = db.query(Employee).filter(
        Employee.is_active == 1,
        Employee.face_encoding.isnot(None),
        Employee.external_admin_id != "",
        Employee.external_user_id != "",
    ).all()
    print(f"Found {len(employees)} employee(s) with a local face to push.")
    for emp in employees:
        print(f"Pushing {emp.name} (id={emp.id})...")
        push_face_to_node(emp.external_admin_id, emp.external_user_id, emp.face_encoding, emp.photo_path)
    print("Done.")
finally:
    db.close()
