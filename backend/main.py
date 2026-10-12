"""
BOTLens face kiosk server.

Recognises faces and reports sightings; keeps nothing about people. Every API
route needs the kiosk's key (`Authorization: Bearer <key>`), which the B.O.T
backend issued when an admin set the kiosk up, and every one of them works only
on that key's company. See hub.py for the backend calls and engine.py for the
face maths.
"""
import base64
import os
from typing import List

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, Request, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

import engine as face  # noqa: E402
import hub  # noqa: E402
import liveness  # noqa: E402
from database import Base, engine as db_engine  # noqa: E402
import models  # noqa: E402,F401  (registers the outbox table)

Base.metadata.create_all(bind=db_engine)

app = FastAPI(title="BOT Lens", docs_url=None, redoc_url=None, openapi_url=None)

# The page is served from this same server, so it needs no CORS at all; these
# are only for local development across ports.
_origins = [o.strip() for o in os.getenv(
    "LENS_ALLOWED_ORIGINS",
    "https://botlens.beontimeofficial.com,http://localhost:8000,https://localhost:8000",
).split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_credentials=False,
                   allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type", "x-enroll-grant"])


@app.exception_handler(Exception)
async def unexpected(request, exc):
    # Logged here, never sent: a stack trace tells a caller how the server is built.
    print(f"--- ERROR {request.method} {request.url.path}: {exc!r} ---")
    return JSONResponse(status_code=500, content={"message": "Something went wrong. Please try again."})


@app.middleware("http")
async def key_first(request: Request, call_next):
    # A request with no kiosk key is refused before anything else is read, so
    # an empty probe gets "not set up", not a form-validation error. The key
    # itself is checked against the backend in each route (hub.whoami).
    if request.url.path.startswith("/api/") and request.method != "OPTIONS" \
            and not (request.headers.get("authorization") or "").strip():
        return JSONResponse(status_code=401, content={"code": "kiosk_key_missing", "message": "This kiosk is not set up."})
    return await call_next(request)


os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("frontend/index.html", headers={"Cache-Control": "no-cache"})


@app.get("/favicon.ico")
def favicon():
    return FileResponse("static/icon.png")


@app.get("/manifest.json")
def manifest():
    return FileResponse("static/manifest.json")


@app.get("/sw.js")
def service_worker():
    return FileResponse("static/sw.js", media_type="application/javascript", headers={"Cache-Control": "no-cache"})


@app.get("/config.js")
def config():
    return Response(content=f'window.NODE_API_URL = "{hub.NODE_API_URL}";', media_type="application/javascript")


@app.get("/healthz")
def healthz():
    return {"ok": True}


def _key(request: Request):
    h = request.headers.get("authorization") or ""
    return h[7:].strip() if h.startswith("Bearer ") else h.strip()


def _refuse(e: hub.BackendError):
    return JSONResponse(status_code=e.status if e.status in (401, 403, 429, 503) else 502, content=e.body)


def _scaled_box(face_row, scale):
    x, y, w, h = [int(round(float(v) * scale)) for v in face_row[:4]]
    return [x, y, w, h]


@app.post("/api/process_client_frame")
def process_client_frame(request: Request, photo: UploadFile = File(...)):
    """
    One camera frame from the kiosk page. Returns the faces found (box, name,
    head pose) and, for each person newly seen, what the backend recorded --
    the kiosk shows that answer, so it can never claim a punch that did not
    happen.
    """
    key = _key(request)
    try:
        me = hub.whoami(key)
    except hub.BackendError as e:
        return _refuse(e)

    img = face.decode_image(photo.file.read())
    if img is None:
        return JSONResponse(status_code=400, content={"message": "The camera frame could not be read."})

    company = str(me["company"]["_id"])
    kiosk = str((me.get("kiosk") or {}).get("_id") or company)
    known = hub.faces_for(key, me)
    work, rows, scale = face.detect(img)
    detections, events = [], []
    # The person in front: the biggest face. The photo check (liveness.py)
    # follows only them.
    primary = max(rows, key=lambda r: float(r[2]) * float(r[3])) if rows else None
    challenge, expired = liveness.current(kiosk)
    if expired:
        hub.release(company, expired["emp_id"])
    for row in rows:
        emp_id, name, score, why = (None, None, 0.0, "unknown")
        pose = face.pose_of(row)
        # Only a near, straight-on face is matched: a small or turned face is
        # where wrong matches come from. The kiosk tells the person what to do.
        if face.face_width(row, scale) < face.MIN_FACE_PX:
            hint = "come_closer"
        elif isinstance(pose.get("rx"), (int, float)) and abs(pose["rx"] - 0.5) > face.MAX_TURN:
            hint = "look_at_camera"
        elif not known:
            hint = "unknown"
        else:
            emp_id, name, score, why = face.best_match(face.embed(work, row), known)
            hint = "matched" if emp_id else why
        det = {"box": _scaled_box(row, scale), "name": name or "Unknown", "pose": pose, "hint": hint}

        # A photo check in progress for the person in front.
        if challenge and row is primary:
            det["name"] = challenge["name"]
            det["challenge"] = challenge["direction"]
            if challenge["stage"] == "turn":
                if liveness.turned_enough(challenge, row):
                    challenge["stage"] = "back"
                    det["hint"] = "turn_back"
                else:
                    det["hint"] = "turn_" + challenge["direction"]
                detections.append(det)
                continue
            # stage "back": straight on again, and still the same person
            if emp_id == challenge["emp_id"]:
                liveness.clear(kiosk)
                result = hub.send_sighting(key, company, emp_id, challenge["score"])
                events.append({"employeeId": emp_id, "name": name, **result})
                challenge = None
            else:
                det["hint"] = "turn_back"
                detections.append(det)
                continue
        elif expired and row is primary and not emp_id:
            det["hint"] = "liveness_retry"

        if emp_id:
            det["employeeId"] = emp_id
            if not challenge and hub.note_match(company, emp_id, me.get("repeatSeconds")):
                if liveness.ENABLED and row is primary:
                    # Recognised: before punching, make sure it is a real head and not a photo.
                    challenge = liveness.start(kiosk, emp_id, name, score, row)
                    det["hint"] = "turn_" + challenge["direction"]
                    det["challenge"] = challenge["direction"]
                elif liveness.ENABLED:
                    hub.release(company, emp_id)  # someone in the background: wait until they are in front
                else:
                    result = hub.send_sighting(key, company, emp_id, score)
                    events.append({"employeeId": emp_id, "name": name, **result})
            last = hub.last_result(company, emp_id)
            det["onDuty"] = bool(last and (last.get("day") or {}).get("onDuty"))
        detections.append(det)
    return {"detections": detections, "events": events}


@app.post("/api/process_frame")
def process_frame(request: Request, file: UploadFile = File(...)):
    """Detection only, for the 360° registration scan: where the face is and how it is turned."""
    try:
        hub.whoami(_key(request))
    except hub.BackendError as e:
        return _refuse(e)
    img = face.decode_image(file.file.read())
    if img is None:
        return {"detections": []}
    _, rows, scale = face.detect(img)
    return {"detections": [{"box": _scaled_box(r, scale), "pose": face.pose_of(r)} for r in rows]}


@app.post("/api/enroll")
def enroll(request: Request, employee_id: str = Form(...), photo: List[UploadFile] = File(...)):
    """
    Register one employee's face. The photos are turned into face numbers here,
    in memory, and only the numbers (plus one small picture for the admin) go to
    the backend. Nothing is written to this server's disk.
    """
    key = _key(request)
    try:
        me = hub.whoami(key)
    except hub.BackendError as e:
        return _refuse(e)

    vectors, thumb_src = [], None
    for p in photo[:12]:
        vec, src = face.encode_photo(p.file.read())
        if vec is not None:
            vectors.append(vec)
            thumb_src = thumb_src or src
    if not vectors:
        return JSONResponse(status_code=400, content={"message": "No face was found in the photos. Try again, facing the camera in good light."})

    # Is this face close to someone ELSE already registered? Saved anyway (two
    # people can look alike), but the admin is told, because the kiosk may then
    # hesitate between them, and a scan saved under the wrong name shows here.
    similar = None
    try:
        others = [f for f in hub.faces_for(key, me) if str(f[0]) != str(employee_id)]
        if others:
            top = face.rank(vectors[0], others)[0]
            if top[0] >= face.MATCH_THRESHOLD - 0.05:
                similar = {"name": top[2], "score": round(top[0], 3)}
    except Exception:
        similar = None

    jpg = face.thumbnail_jpeg(thumb_src)
    thumb = "data:image/jpeg;base64," + base64.b64encode(jpg).decode() if jpg else None
    try:
        saved = hub.save_face(key, request.headers.get("x-enroll-grant"), employee_id, vectors, thumb)
    except hub.BackendError as e:
        return JSONResponse(status_code=e.status, content=e.body)
    except Exception:
        return JSONResponse(status_code=503, content={"message": "Cannot reach the B.O.T server. The face was not saved."})
    hub.drop_faces(me["company"]["_id"])
    hub.forget_key(key)  # next frame re-reads the faces version
    return {"ok": True, "scans": len(vectors), "employee": saved.get("employee"), "similarTo": similar}


@app.get("/api/status")
def status(request: Request):
    """For the kiosk page's footer: is this kiosk set up, and is anything waiting to be sent."""
    try:
        me = hub.whoami(_key(request))
    except hub.BackendError as e:
        return _refuse(e)
    return {"company": me.get("company"), "kiosk": me.get("kiosk"), "waitingToSend": hub.outbox_size()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000, ssl_keyfile="certs/key.pem", ssl_certfile="certs/cert.pem")
