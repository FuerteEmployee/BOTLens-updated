"""
Everything the kiosk needs from the B.O.T backend, and nothing it keeps.

  · who a kiosk key belongs to   GET  /api/lens/me      (cached 60 s per key)
  · that company's faces         GET  /api/lens/faces   (reloaded when the
                                                         backend's version moves)
  · a sighting                   POST /api/lens/taps    (the answer is what the
                                                         kiosk shows)
  · a face registration          PUT  /api/lens/faces/:id

Every call carries the kiosk's own key (sent by the kiosk page), so the
backend decides which company it is, never this server. Faces of one company
are never matched against another company's camera.
"""
import hashlib
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import requests

from database import SessionLocal
from models import OutboxTap

NODE_API_URL = os.getenv("NODE_API_URL", "http://localhost:5000").rstrip("/")
TIMEOUT = float(os.getenv("NODE_TIMEOUT_SECONDS", "6"))
ME_TTL = 60            # re-ask the backend who a key is, at most once a minute
ME_STALE_OK = 600      # keep working on the last answer this long if the backend is down
SEEN_FRAMES = 2        # a face must match in this many frames ...
SEEN_WINDOW = 3.0      # ... within this many seconds before it counts as a sighting


class BackendError(Exception):
    def __init__(self, status, body):
        super().__init__(body.get("message") if isinstance(body, dict) else str(body))
        self.status = status
        self.body = body if isinstance(body, dict) else {"message": str(body)}


def _hash(key):
    return hashlib.sha256(key.encode()).hexdigest()


def _call(method, path, key, json_body=None, extra_headers=None):
    headers = {"Authorization": f"Bearer {key}"}
    if extra_headers:
        headers.update(extra_headers)
    r = requests.request(method, f"{NODE_API_URL}/api{path}", json=json_body, headers=headers, timeout=TIMEOUT)
    try:
        body = r.json()
    except ValueError:
        body = {"message": r.text[:200]}
    if r.status_code >= 400:
        raise BackendError(r.status_code, body)
    return body


# ── who is this kiosk ─────────────────────────────────────────────────────────
_me_cache = {}  # key hash -> (fetched_at, body)
_me_lock = threading.Lock()


def whoami(key):
    """
    The backend's answer for this key: company, kiosk, faces version, how long
    a repeat sighting is ignored. A rejected key raises BackendError (401/403).
    If the backend cannot be reached, the last answer is used for up to 10 min.
    """
    if not key:
        raise BackendError(401, {"code": "kiosk_key_missing", "message": "This kiosk is not set up."})
    h = _hash(key)
    now = time.time()
    with _me_lock:
        hit = _me_cache.get(h)
    if hit and now - hit[0] < ME_TTL:
        return hit[1]
    try:
        body = _call("GET", "/lens/me", key)
    except BackendError:
        with _me_lock:
            _me_cache.pop(h, None)
        raise
    except requests.RequestException:
        if hit and now - hit[0] < ME_STALE_OK:
            return hit[1]
        raise BackendError(503, {"message": "Cannot reach the B.O.T server. Check the internet connection."})
    with _me_lock:
        _me_cache[h] = (now, body)
    return body


def forget_key(key):
    with _me_lock:
        _me_cache.pop(_hash(key), None)


# ── faces, per company ────────────────────────────────────────────────────────
_faces = {}  # company id -> {"version": str|None, "items": [(emp_id, name, [vec...])]}
_faces_lock = threading.Lock()


def faces_for(key, me):
    """This company's faces, reloaded only when the backend says they changed."""
    company = str(me["company"]["_id"])
    version = me.get("facesVersion")
    with _faces_lock:
        cached = _faces.get(company)
    if cached and cached["version"] == version:
        return cached["items"]
    try:
        body = _call("GET", "/lens/faces", key)
    except requests.RequestException:
        return cached["items"] if cached else []
    items = []
    for f in body.get("faces", []):
        vectors = [np.array(v, dtype=np.float32) for v in f.get("embeddings", []) if len(v) == 128]
        if vectors:
            items.append((str(f["employeeId"]), f.get("name") or "", vectors))
    with _faces_lock:
        _faces[company] = {"version": body.get("version"), "items": items}
    print(f"--- faces loaded for company {company[-6:]}: {len(items)} ---")
    return items


def drop_faces(company):
    with _faces_lock:
        _faces.pop(str(company), None)


# ── sightings ─────────────────────────────────────────────────────────────────
# Per company+employee: recent matching frames, whether a tap is in flight, and
# when the next sighting may be sent. Display state only: what the sighting
# MEANS is decided by the backend.
_seen = {}
_seen_lock = threading.Lock()


def note_match(company, emp_id, repeat_seconds):
    """
    Record one matching frame. Returns True when this is a sighting to send:
    seen in enough frames, nothing in flight, and past the repeat window of the
    last one sent (the backend's double-tap window; it would refuse it anyway).
    """
    k = (str(company), str(emp_id))
    now = time.time()
    with _seen_lock:
        s = _seen.setdefault(k, {"hits": [], "inflight": False, "next_ok": 0.0, "last": None})
        s["hits"] = [t for t in s["hits"] if now - t <= SEEN_WINDOW] + [now]
        if s["inflight"] or now < s["next_ok"] or len(s["hits"]) < SEEN_FRAMES:
            return False
        s["inflight"] = True
        s["next_ok"] = now + max(30, repeat_seconds or 120)
        return True


def last_result(company, emp_id):
    with _seen_lock:
        s = _seen.get((str(company), str(emp_id)))
        return s["last"] if s else None


def _finish(company, emp_id, result, retry_after=None):
    with _seen_lock:
        s = _seen.get((str(company), str(emp_id)))
        if not s:
            return
        s["inflight"] = False
        if result is not None:
            s["last"] = result
        if retry_after:
            s["next_ok"] = time.time() + retry_after


def send_sighting(key, company, emp_id, score):
    """
    Report one sighting and return what the backend made of it, for the kiosk
    to show. If the backend cannot be reached, the sighting is kept in the
    outbox with its real time and sent later.
    """
    tap_time = datetime.now(timezone.utc).isoformat()
    payload = {"employeeId": emp_id, "tapTime": tap_time, "score": round(float(score), 4)}
    try:
        body = _call("POST", "/lens/taps", key, payload)
        _finish(company, emp_id, body, body.get("retryAfterSec"))
        return body
    except BackendError as e:
        _finish(company, emp_id, None)
        return {"recorded": False, "reason": "refused", "message": e.body.get("message") or "Not recorded.", "status": e.status}
    except requests.RequestException:
        _queue(key, payload)
        _finish(company, emp_id, None)
        return {"recorded": False, "reason": "offline", "offline": True,
                "message": "Saved. It will be sent when the connection is back."}


def _queue(key, payload):
    db = SessionLocal()
    try:
        now = time.time()
        db.add(OutboxTap(event_id=str(uuid.uuid4()), kiosk_key=key, payload=json.dumps(payload),
                         attempts=0, next_try=now + 15, created_at=now))
        db.commit()
    finally:
        db.close()


def _outbox_worker():
    while True:
        time.sleep(15)
        db = SessionLocal()
        try:
            due = db.query(OutboxTap).filter(OutboxTap.next_try <= time.time()).limit(50).all()
            for row in due:
                try:
                    _call("POST", "/lens/taps", row.kiosk_key, json.loads(row.payload))
                    db.delete(row)  # recorded, or a duplicate of one already recorded
                except BackendError as e:
                    if e.status >= 500 or e.status == 429:
                        row.attempts += 1
                        row.next_try = time.time() + min(900, 30 * row.attempts)
                    else:
                        print(f"--- outbox: dropped a sighting the backend refused ({e.status}) ---")
                        db.delete(row)
                except requests.RequestException:
                    row.attempts += 1
                    row.next_try = time.time() + min(900, 30 * row.attempts)
            db.commit()
        except Exception as e:  # the worker must never die
            print(f"--- outbox worker error: {e} ---")
        finally:
            db.close()


threading.Thread(target=_outbox_worker, daemon=True).start()


def outbox_size():
    db = SessionLocal()
    try:
        return db.query(OutboxTap).count()
    finally:
        db.close()


# ── registration ──────────────────────────────────────────────────────────────
def save_face(key, grant, employee_id, vectors, thumbnail_data_url):
    return _call("PUT", f"/lens/faces/{employee_id}", key,
                 {"embeddings": [v.tolist() for v in vectors], "thumbnail": thumbnail_data_url,
                  "modelVersion": "sface_2021dec"},
                 {"x-enroll-grant": grant or ""})
