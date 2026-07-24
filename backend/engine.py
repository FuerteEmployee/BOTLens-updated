import cv2
import json
import time
import os
import threading
import queue
import requests
import numpy as np
from sqlalchemy.orm import Session
from models import Employee, AttendanceLog, ActivityStats
from datetime import date, datetime, timedelta
from database import SessionLocal

# Optional sync to the Node/HRMS backend so camera-detected punches show up
# there too. Left unset, this is a no-op (see sync_punch_to_node).
NODE_API_URL = os.getenv("NODE_API_URL", "").rstrip("/")
NODE_CAMERA_API_KEY = os.getenv("NODE_CAMERA_API_KEY", "")

# Node's punch handlers do a read-modify-save on the same Attendance document
# (not an atomic update), so firing one thread per event let concurrent
# requests race and silently lose updates to each other. A single FIFO queue
# processed by one worker thread guarantees every event reaches Node exactly
# once, in the order it actually happened.
_sync_queue = queue.Queue()

def _sync_worker():
    while True:
        admin_id, employee_id, action = _sync_queue.get()
        try:
            requests.post(
                f"{NODE_API_URL}/api/device/attendance/punch",
                json={"adminId": admin_id, "employeeId": employee_id, "action": action},
                headers={"Authorization": f"Bearer {NODE_CAMERA_API_KEY}"},
                timeout=5,
            )
        except Exception as e:
            print(f"--- NODE SYNC ERROR: {str(e)} ---")
        finally:
            _sync_queue.task_done()

threading.Thread(target=_sync_worker, daemon=True).start()

def sync_punch_to_node(admin_id: str, employee_id: str, action: str):
    if not NODE_API_URL or not NODE_CAMERA_API_KEY:
        return
    _sync_queue.put((admin_id, employee_id, action))

ai_error = None
try:
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    det_model = os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")
    rec_model = os.path.join(models_dir, "face_recognition_sface_2021dec.onnx")
    
    if os.path.exists(det_model) and os.path.exists(rec_model):
        face_detector = cv2.FaceDetectorYN.create(det_model, "", (320, 320), 0.6, 0.3, 5000)
        face_recognizer = cv2.FaceRecognizerSF.create(rec_model, "")
        use_deep_ai = True
        print(f"--- SUCCESS: YuNet + SFace Active at {models_dir} ---")
    else:
        use_deep_ai = False
        ai_error = f"Models missing at {models_dir}. Files found: {os.listdir(models_dir) if os.path.exists(models_dir) else 'None'}"
        print(f"--- WARNING: {ai_error} ---")
except Exception as e:
    use_deep_ai = False
    ai_error = str(e)
    print(f"--- ERROR loading Deep AI: {ai_error} ---")

# Fallback Haarcascade
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception as e:
    print(f"--- ERROR loading fallback: {str(e)} ---")

# Memory Cache for Recognition
known_face_encodings = {} # {emp_id: [vector]}
needs_retrain = True 

presence_state = {} # {emp_id: {'status': 'IN'/'OUT', 'last_action': float}}
COOLDOWN_SECONDS = 5 
event_messages = [] 
last_recognition_time = 0

def trigger_retrain():
    global needs_retrain
    needs_retrain = True

def train_recognizer(db: Session):
    global known_face_encodings, needs_retrain
    employees = db.query(Employee).filter(Employee.is_active == 1).all()
    known_face_encodings = {}
    
    for emp in employees:
        try:
            emp_feats = []
            # 1. Check if we have multiple photos in the directory
            if emp.photo_path:
                target_dir = os.path.dirname(emp.photo_path)
                if os.path.isdir(target_dir) and "data/photos" in target_dir:
                    files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    for f in files:
                        feat = encode_face(f)
                        if feat is not None: emp_feats.append(feat)
            
            # 2. Fallback to DB encoding if no photos or extra angles needed
            if not emp_feats and emp.face_encoding:
                try:
                    data = json.loads(emp.face_encoding)
                    if isinstance(data, list):
                        # Could be a single feature or list of features
                        if isinstance(data[0], (int, float)):
                            emp_feats.append(np.array(data, dtype=np.float32))
                        else:
                            for d in data: emp_feats.append(np.array(d, dtype=np.float32))
                except: pass

            if emp_feats:
                known_face_encodings[emp.id] = emp_feats
                
        except Exception as e:
            print(f"Error loading encoding for {emp.name}: {e}")
            
    needs_retrain = False
    print(f"--- Loaded encodings for {len(known_face_encodings)} employees ---")
    return True

def encode_face(image_path: str):
    if not os.path.exists(image_path): return None
    img = cv2.imread(image_path)
    if img is None: return None
    
    if use_deep_ai:
        try:
            # Try with different input sizes for better detection at high-res
            h, w = img.shape[:2]
            face_detector.setInputSize((w, h))
            
            # Use a slightly lower score threshold for the final photo encoding
            # to be extra sure we don't block the user
            _, faces = face_detector.detect(img)
            
            if faces is not None and len(faces) > 0:
                aligned_face = face_recognizer.alignCrop(img, faces[0])
                feature = face_recognizer.feature(aligned_face)
                return feature[0]
            else:
                # Fallback: Try resizing to 640p for detection if original was too large/small
                scale = 640 / max(h, w)
                resized = cv2.resize(img, (0,0), fx=scale, fy=scale)
                face_detector.setInputSize((resized.shape[1], resized.shape[0]))
                _, faces = face_detector.detect(resized)
                if faces is not None and len(faces) > 0:
                    aligned_face = face_recognizer.alignCrop(resized, faces[0])
                    feature = face_recognizer.feature(aligned_face)
                    return feature[0]
                    
        except Exception as e:
            print(f"SFace Encoding Error: {e}")
    return None

def get_known_faces(db: Session):
    employees = db.query(Employee).filter(Employee.is_active == 1).all()
    # Return a map for easy lookup
    return {emp.id: emp.name for emp in employees}

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_or_create_daily_stats(db: Session, emp_id: int):
    today = date.today().strftime("%Y-%m-%d")
    stats = db.query(ActivityStats).filter(ActivityStats.employee_id == emp_id, ActivityStats.date == today).first()
    if not stats:
        stats = ActivityStats(employee_id=emp_id, date=today, work_seconds=0, phone_seconds=0)
        db.add(stats); db.commit(); db.refresh(stats)
    return stats

def detect_pose(det):
    # Landmarks: [lex, ley, rex, rey, nx, ny, mlx, mly, mrx, mry]
    try:
        # Note: Frontend cameras are usually mirrored. 
        # lex is the landmark appearing on the left of the image.
        lex, ley = det[4], det[5]
        rex, rey = det[6], det[7]
        nx, ny = det[8], det[9]
        
        # Yaw (Horizontal rotation)
        # We use the horizontal position of the nose relative to the eyes.
        eye_dist = max(1, abs(rex - lex))
        # ratio_x: 0.5 is center, < 0.5 is towards lex, > 0.5 is towards rex
        ratio_x = (nx - min(lex, rex)) / eye_dist
        
        # Pitch (Vertical rotation)
        eye_y_avg = (ley + rey) / 2
        ratio_y = (ny - eye_y_avg) / eye_dist 
        
        # Mapping to directions (adjusted for typical mirrored front-cam)
        yaw = "CENTER"
        if ratio_x < 0.38: yaw = "RIGHT"   # Nose closer to image-left (user's right)
        elif ratio_x > 0.62: yaw = "LEFT"  # Nose closer to image-right (user's left)
        
        pitch = "CENTER"
        if ratio_y < 0.28: pitch = "UP"
        elif ratio_y > 0.58: pitch = "DOWN"
        
        return {"yaw": yaw, "pitch": pitch, "rx": round(float(ratio_x), 2), "ry": round(float(ratio_y), 2)}
    except Exception as e:
        return {"yaw": "UNKNOWN", "pitch": "UNKNOWN"}

def process_frame(frame, db: Session, detections_list=None, recognition=True):
    global presence_state, last_recognition_time, event_messages, needs_retrain, known_face_encodings
    if db: db.expire_all()
    
    current_time = time.time()
    
    # 0. Periodic Retrain OR Manual Trigger
    if needs_retrain or current_time - last_recognition_time > 300: 
        train_recognizer(db)
        last_recognition_time = current_time

    # Cleanup old event messages (show for 3 seconds)
    event_messages = [m for m in event_messages if current_time - m['time'] < 3]

    h_frame, w_frame = frame.shape[:2]
    
    # 1. Detection & Recognition
    raw_detections = []
    if use_deep_ai:
        # Optimization: If frame is very small, upscale it for better YuNet detection
        if w_frame < 400:
            frame = cv2.resize(frame, (0,0), fx=2, fy=2)
            w_frame, h_frame = frame.shape[1], frame.shape[0]

        face_detector.setInputSize((w_frame, h_frame))
        _, raw_detections = face_detector.detect(frame)
    else:
        # Fallback to Haarcascade
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_found = face_cascade.detectMultiScale(gray_frame, 1.03, 3, minSize=(20, 20))
        for (x, y, w, h) in faces_found:
            d = np.zeros(15, dtype=np.float32)
            d[0:4] = [x, y, w, h]
            raw_detections.append(d)
    
    known_map = get_known_faces(db)
    
    try:
        if raw_detections is not None:
            for det in raw_detections:
                x, y, w, h = map(int, det[0:4])
                emp_id = None
                emp_name = "Unknown"
                
                if recognition and use_deep_ai and len(known_face_encodings) > 0:
                    try:
                        aligned_face = face_recognizer.alignCrop(frame, det)
                        query_feat = face_recognizer.feature(aligned_face)[0]
                        
                        best_id = None
                        best_score = -1
                        for eid, feat_list in known_face_encodings.items():
                            # Compare against all angles captured for this employee
                            for target_feat in feat_list:
                                score = face_recognizer.match(query_feat, target_feat, cv2.FaceRecognizerSF_FR_COSINE)
                                if score > best_score:
                                    best_score = score
                                    best_id = eid
                        
                        # SFace Threshold: 0.36 is standard, 0.45 is strict
                        if best_score > 0.42: # Slightly relaxed for side angles
                            emp_id = best_id
                            emp_name = known_map.get(emp_id, "Unknown")
                    except Exception as e:
                        print(f"Recognition Error: {e}")
                
                status_text = "OUT"
                if emp_id:
                    state = presence_state.get(emp_id, {'status': 'OUT', 'last_action': 0})
                    if (current_time - state['last_action']) > COOLDOWN_SECONDS:
                        # Simplified Smart Retroactive Logic
                        today_start = datetime.combine(date.today(), datetime.min.time())
                        
                        if state['status'] == "OUT":
                            # Next is an IN action
                            # 1. Retroactively change previous PUNCH_OUT to LUNCH_OUT
                            last_log = db.query(AttendanceLog).filter(
                                AttendanceLog.employee_id == emp_id, 
                                AttendanceLog.timestamp >= today_start
                            ).order_by(AttendanceLog.timestamp.desc()).first()
                            
                            if last_log and last_log.action == "PUNCH_OUT":
                                last_log.action = "LUNCH_OUT"
                            
                            # 2. 1st In of the day is PUNCH_IN, all others are LUNCH_IN
                            has_logs = db.query(AttendanceLog).filter(
                                AttendanceLog.employee_id == emp_id, 
                                AttendanceLog.timestamp >= today_start
                            ).first() is not None
                            
                            new_action = "PUNCH_IN" if not has_logs else "LUNCH_IN"
                            new_status = "IN"
                        else:
                            # Next is an OUT action - Always start as PUNCH_OUT
                            new_action = "PUNCH_OUT"
                            new_status = "OUT"

                        db.add(AttendanceLog(employee_id=emp_id, action=new_action, reason="FACE_MATCH"))
                        if new_status == "OUT" and state['last_action'] > 0:
                            duration = int(current_time - state['last_action'])
                            stats = get_or_create_daily_stats(db, emp_id)
                            stats.work_seconds += duration
                        db.commit()

                        emp_obj = db.query(Employee).filter(Employee.id == emp_id).first()
                        if emp_obj and emp_obj.external_admin_id and emp_obj.external_user_id:
                            # Node's lunch-in/out is a one-time-per-day sub-event,
                            # not a repeatable toggle — every arrival syncs as
                            # punch-in and every departure as punch-out so Node's
                            # own multi-session (shifts array) logic handles the
                            # re-entries, regardless of BOTLens's local
                            # PUNCH_IN/LUNCH_IN labeling (used only for its own UI).
                            node_sync_action = "punch-in" if new_status == "IN" else "punch-out"
                            sync_punch_to_node(emp_obj.external_admin_id, emp_obj.external_user_id, node_sync_action)

                        presence_state[emp_id] = {'status': new_status, 'last_action': current_time}
                        event_messages.append({'name': emp_name, 'action': new_action.replace('_', ' '), 'time': current_time})
                    status_text = presence_state[emp_id]['status']

                    # Draw for fallback compatibility
                    color = (0, 255, 0) if status_text == "IN" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                    cv2.putText(frame, f"{emp_name} ({status_text})", (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Populate list for high-speed client rendering
                if detections_list is not None:
                    # If Deep AI is active, get real pose. Else provide a 'CENTER' fallback 
                    # so the user can at least proceed with the scan.
                    if use_deep_ai:
                        pose = detect_pose(det)
                    else:
                        # Standard AI Fallback: Always return CENTER so scan can proceed
                        pose = {"yaw": "CENTER", "pitch": "CENTER"}
                        
                    detections_list.append({
                        "box": [x, y, w, h],
                        "name": emp_name,
                        "status": status_text,
                        "pose": pose
                    })

        for i, msg in enumerate(event_messages):
            text = f"{msg['name']} {msg['action']} Successful"
            cv2.putText(frame, text, (50, 50 + (i * 40)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            cv2.putText(frame, text, (50, 50 + (i * 40)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

    except Exception as e:
        print(f"--- ENGINE ERROR: {str(e)} ---")
    
    return frame

    # 3. Punch Out Logic (Independent)
    for eid, state in list(presence_state.items()):
        if state["status"] == "IN" and (current_time - state["last_seen"]) > TIMEOUT_SECONDS:
            state["status"] = "OUT"
            db.add(AttendanceLog(employee_id=eid, action="PUNCH_OUT", reason="LEFT_DESK"))
            db.commit()
            if eid in activity_timers: del activity_timers[eid]

    return frame
