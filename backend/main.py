import os
from dotenv import load_dotenv
load_dotenv()
import cv2
import json
import time
import numpy as np
import base64
from fastapi import FastAPI, Depends, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from database import engine, Base, get_db
from models import Employee, AttendanceLog, ActivityStats
from engine import process_frame, encode_face, trigger_retrain
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse, FileResponse
import uvicorn

from typing import List

# Create tables after models are loaded
Base.metadata.create_all(bind=engine)

print("--- BOTLens v1.2 Deployment (Distance Opt + Upload) Active ---")
app = FastAPI(title="BOT Lens")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    import traceback
    err = traceback.format_exc()
    print(f"--- GLOBAL ERROR STACK: {err} ---")
    return JSONResponse(status_code=500, content={"message": str(exc), "trace": err})

photo_dir = os.getenv("PHOTO_DIR", "data/photos")
os.makedirs(photo_dir, exist_ok=True)
app.mount("/photos", StaticFiles(directory=photo_dir), name="photos")

def photo_url(path: str):
    if not path: return None
    rel = os.path.relpath(path, photo_dir).replace(os.sep, "/")
    return f"/photos/{rel}"
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/icon.png")

@app.get("/manifest.json")
async def get_manifest():
    return FileResponse("static/manifest.json")

@app.get("/config.js")
async def get_config():
    node_api_url = os.getenv("NODE_API_URL", "http://localhost:5000")
    return Response(content=f'window.NODE_API_URL = "{node_api_url}";', media_type="application/javascript")

@app.get("/sw.js")
async def get_sw():
    return FileResponse("static/sw.js", media_type="application/javascript")

# Distributed Frame Processing (Every device uses its own camera)
@app.post("/api/process_client_frame")
async def process_client_frame(
    photo: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    contents = await photo.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return {"error": "Invalid image"}

    # Process using engine
    try:
        # Get detections and names instead of a full processed image
        detections = [] # List of {x, y, w, h, name, status}
        processed_frame = process_frame(frame, db, detections_list=detections)
        
        _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "image": encoded_image,
            "detections": detections
        }
    except Exception as e:
        print(f"--- FRAME ERROR: {str(e)} ---")
        return {"error": str(e)}

@app.post("/api/employees")
async def add_employee(
    name: str = Form(...),
    photo: List[UploadFile] = File(...),
    phone: str = Form(""),
    monthly_salary: float = Form(0.0),
    hourly_rate: float = Form(0.0),
    bank_name: str = Form(""),
    bank_account: str = Form(""),
    bank_ifsc: str = Form(""),
    external_admin_id: str = Form(""),
    external_user_id: str = Form(""),
    db: Session = Depends(get_db)
):
    # Create employee-specific folder
    timestamp = int(time.time())
    emp_folder = os.path.join(photo_dir, f"{name.replace(' ', '_')}_{timestamp}")
    os.makedirs(emp_folder, exist_ok=True)
    
    saved_paths = []
    for i, p in enumerate(photo):
        p_path = os.path.join(emp_folder, f"face_{i}.jpg")
        with open(p_path, "wb") as b: b.write(await p.read())
        saved_paths.append(p_path)
    
    if not saved_paths:
        return {"error": "No photos received"}

    # Check first photo for face
    enc = encode_face(saved_paths[0])
    if enc is None:
        import shutil
        shutil.rmtree(emp_folder)
        return {"error": "No face detected in primary photo. Please try again."}
    
    emp = Employee(
        name=name, phone=phone, photo_path=saved_paths[0], face_encoding=json.dumps(enc.tolist()),
        monthly_salary=monthly_salary, hourly_rate=hourly_rate,
        bank_name=bank_name, bank_account=bank_account, bank_ifsc=bank_ifsc,
        external_admin_id=external_admin_id, external_user_id=external_user_id
    )
    db.add(emp); db.commit(); db.refresh(emp)
    trigger_retrain()
    return {"id": emp.id}

@app.get("/api/employees")
def get_employees(external_admin_id: str = None, db: Session = Depends(get_db)):
    query = db.query(Employee).filter(Employee.is_active == 1)
    if external_admin_id:
        query = query.filter(Employee.external_admin_id == external_admin_id)
    emps = query.all()
    return [{
        "id": e.id, "name": e.name, "phone": e.phone, "photo_path": e.photo_path, "photo_url": photo_url(e.photo_path),
        "monthly_salary": e.monthly_salary, "hourly_rate": e.hourly_rate,
        "bank_name": e.bank_name, "bank_account": e.bank_account, "bank_ifsc": e.bank_ifsc,
        "external_admin_id": e.external_admin_id, "external_user_id": e.external_user_id
    } for e in emps]

@app.put("/api/employees/{emp_id}")
async def update_employee(
    emp_id: int,
    name: str = Form(...),
    phone: str = Form(None),
    monthly_salary: float = Form(None),
    hourly_rate: float = Form(None),
    bank_name: str = Form(None),
    bank_account: str = Form(None),
    bank_ifsc: str = Form(None),
    external_admin_id: str = Form(None),
    external_user_id: str = Form(None),
    photo: List[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    emp = db.query(Employee).filter(Employee.id == emp_id).first()
    if emp:
        emp.name = name
        if phone is not None: emp.phone = phone
        if monthly_salary is not None: emp.monthly_salary = monthly_salary
        if hourly_rate is not None: emp.hourly_rate = hourly_rate
        if bank_name is not None: emp.bank_name = bank_name
        if bank_account is not None: emp.bank_account = bank_account
        if bank_ifsc is not None: emp.bank_ifsc = bank_ifsc
        if external_admin_id is not None: emp.external_admin_id = external_admin_id
        if external_user_id is not None: emp.external_user_id = external_user_id

        # Replace the face photo(s) — re-derive the encoding so recognition
        # picks up the new face immediately after retrain. Written to a
        # staging folder first and validated *before* touching the employee's
        # existing photos, so a bad upload (no face detected) can't destroy
        # the current working photo.
        if photo:
            emp_folder = os.path.dirname(emp.photo_path) if emp.photo_path else os.path.join(photo_dir, f"{name.replace(' ', '_')}_{emp.id}")
            staging_folder = emp_folder + "_staging"
            import shutil
            if os.path.isdir(staging_folder): shutil.rmtree(staging_folder)
            os.makedirs(staging_folder, exist_ok=True)

            staged_paths = []
            for i, p in enumerate(photo):
                p_path = os.path.join(staging_folder, f"face_{i}.jpg")
                with open(p_path, "wb") as b: b.write(await p.read())
                staged_paths.append(p_path)

            if staged_paths:
                enc = encode_face(staged_paths[0])
                if enc is None:
                    shutil.rmtree(staging_folder)
                    return {"error": "No face detected in the new photo. Please try again."}

                # Validated — now safe to replace the live folder's contents.
                os.makedirs(emp_folder, exist_ok=True)
                for f in os.listdir(emp_folder):
                    if f.startswith("face_"): os.remove(os.path.join(emp_folder, f))
                saved_paths = []
                for i, p in enumerate(staged_paths):
                    dest = os.path.join(emp_folder, f"face_{i}.jpg")
                    shutil.move(p, dest)
                    saved_paths.append(dest)
                shutil.rmtree(staging_folder, ignore_errors=True)

                emp.photo_path = saved_paths[0]
                emp.face_encoding = json.dumps(enc.tolist())

        db.commit()
        trigger_retrain()
    return {"success": True}

@app.delete("/api/employees/{emp_id}")
async def delete_employee(emp_id: int, db: Session = Depends(get_db)):
    emp = db.query(Employee).filter(Employee.id == emp_id).first()
    if emp: 
        emp.is_active = 0
        db.commit()
        trigger_retrain()
    return {"success": True}

@app.get("/api/history")
def get_history(emp_id: int = None, date_str: str = None, external_admin_id: str = None, db: Session = Depends(get_db)):
    query = db.query(AttendanceLog).join(Employee).filter(Employee.is_active == 1)
    if emp_id:
        query = query.filter(AttendanceLog.employee_id == emp_id)
    if external_admin_id:
        query = query.filter(Employee.external_admin_id == external_admin_id)
    if date_str:
        # date format: YYYY-MM-DD
        from datetime import datetime
        start_date = datetime.strptime(date_str, "%Y-%m-%d")
        end_date = start_date.replace(hour=23, minute=59, second=59)
        query = query.filter(AttendanceLog.timestamp >= start_date, AttendanceLog.timestamp <= end_date)
    
    logs = query.order_by(AttendanceLog.timestamp.asc()).all()
    return [{"id": l.id, "employee_name": l.employee.name if l.employee else "Unknown", "action": l.action, "reason": l.reason, "timestamp": l.timestamp.strftime("%Y-%m-%d %H:%M:%S")} for l in logs]

@app.get("/api/stats")
def get_stats(external_admin_id: str = None, db: Session = Depends(get_db)):
    query = db.query(ActivityStats).join(Employee).filter(Employee.is_active == 1)
    if external_admin_id:
        query = query.filter(Employee.external_admin_id == external_admin_id)
    stats = query.order_by(ActivityStats.date.desc()).all()
    res = []
    for s in stats:
        h_w, m_w, s_w = s.work_seconds // 3600, (s.work_seconds % 3600) // 60, s.work_seconds % 60
        h_p, m_p, s_p = s.phone_seconds // 3600, (s.phone_seconds % 3600) // 60, s.phone_seconds % 60
        daily_salary = 0.0
        emp = s.employee
        if emp.monthly_salary > 0: daily_salary = emp.monthly_salary / 30.0
        elif emp.hourly_rate > 0: daily_salary = (s.work_seconds / 3600.0) * emp.hourly_rate
        res.append({
            "employee_name": emp.name, "date": s.date,
            "work_time": f"{h_w:02d}:{m_w:02d}:{s_w:02d}",
            "phone_time": f"{h_p:02d}:{m_p:02d}:{s_p:02d}",
            "total_seconds": s.work_seconds + s.phone_seconds,
            "daily_salary": round(daily_salary, 2)
        })
    return res

@app.get("/")
def get_index(): return FileResponse("frontend/index.html")

@app.post("/api/process_frame")
async def api_process_frame(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None: return {"detections": []}
        
        detections = []
        # Run process_frame WITHOUT recognition (detection only)
        process_frame(frame, db, detections_list=detections, recognition=False)
        
        return {"detections": detections}
    except Exception as e:
        print(f"API Process Frame Error: {e}")
        return {"detections": [], "error": str(e)}

if __name__ == "__main__": 
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000, ssl_keyfile="certs/key.pem", ssl_certfile="certs/cert.pem")
