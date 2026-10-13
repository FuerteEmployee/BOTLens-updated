"""
Face detection and matching only.

The kiosk keeps nothing about people: faces live in the B.O.T backend
(FaceProfile) and are loaded per company by hub.py; what a sighting MEANS is
decided by the backend (a sighting is a tap, read in one day timeline with the
app and the fingerprint machine). This file only turns pixels into numbers and
numbers into "best match".

Models (backend/models/): YuNet face detector + SFace recognizer, OpenCV's own.
An SFace vector is 128 floats; two faces match by cosine similarity.
"""
import math
import os
import threading

import cv2
import numpy as np

MODEL_VERSION = "sface_2021dec"
# How a person is scored, and when a match counts. Chosen on the registered
# faces themselves (12 Oct, staging and production): with "best single angle
# >= 0.42" an unregistered person's straight-on face was taken for someone
# else in 3 of 6 cases (Aryan seen as Nihal before Aryan was registered).
# Scoring a person as half their straight-on angle + half their best angle,
# and requiring 0.45, took that to 0 of 6 (production 0 of 4) while every
# registered person's straight-on face was still accepted as before.
MATCH_THRESHOLD = float(os.getenv("LENS_MATCH_THRESHOLD", "0.45"))
# Two registered people this close to each other for one face: say nothing
# rather than guess. A wrong match punches the wrong person.
MATCH_MARGIN = float(os.getenv("LENS_MATCH_MARGIN", "0.05"))
# A face narrower than this (pixels in the camera frame) is too far away to
# recognise reliably; the kiosk asks the person to come closer instead.
MIN_FACE_PX = int(os.getenv("LENS_MIN_FACE_PX", "70"))
# How far a head may be turned and still be matched: 0.5 is straight on (nose
# between the eyes), 0.5 +/- this is allowed. Wider than the registration
# scan's "LOOK CENTER" band, since people walk up to a kiosk at an angle;
# a face in strong profile is where wrong matches come from.
MAX_TURN = float(os.getenv("LENS_MAX_TURN", "0.22"))

_models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
_det_path = os.path.join(_models_dir, "face_detection_yunet_2023mar.onnx")
_rec_path = os.path.join(_models_dir, "face_recognition_sface_2021dec.onnx")

# No fallback detector. The Haar cascade this used to fall back to cannot
# recognise anyone, and with OpenCV 5 it loads empty without raising: a kiosk
# that silently sees no faces is worse than one that refuses to start.
if not (os.path.exists(_det_path) and os.path.exists(_rec_path)):
    raise RuntimeError(f"Face models missing in {_models_dir}: need YuNet and SFace .onnx files")

_detector = cv2.FaceDetectorYN.create(_det_path, "", (320, 320), 0.6, 0.3, 5000)
_recognizer = cv2.FaceRecognizerSF.create(_rec_path, "")

# One detector and one recognizer serve every request, and FastAPI runs each
# request on its own thread. YuNet's input size is per-object state: two
# frames of different sizes at once (a live frame and a 360° scan frame, or two
# kiosks) set it under each other, and OpenCV then failed with
# "buf.shape() == m.shape()" or returned boxes of infinity (316 such errors on
# the staging kiosk, 12 Oct). Every use goes through this lock.
_cv_lock = threading.Lock()
print(f"--- Face models loaded from {_models_dir} ---")


def decode_image(data: bytes):
    """JPEG/PNG bytes -> BGR image, or None."""
    if not data:
        return None
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def detect(img):
    """All faces in an image, as YuNet rows (box, 5 landmarks, score)."""
    h, w = img.shape[:2]
    if w < 400:  # small frames: upscale so a face at a distance is still found
        img = cv2.resize(img, (0, 0), fx=2, fy=2)
        h, w = img.shape[:2]
        scale = 0.5
    else:
        scale = 1.0
    with _cv_lock:
        _detector.setInputSize((w, h))
        _, faces = _detector.detect(img)
    rows = [] if faces is None else [f for f in faces if all(math.isfinite(float(v)) for v in f[:15])]
    return img, rows, scale


def embed(img, face_row):
    """128-value SFace vector for one detected face."""
    with _cv_lock:
        aligned = _recognizer.alignCrop(img, face_row)
        return _recognizer.feature(aligned)[0].copy()


def encode_photo(data: bytes):
    """
    One registration photo -> (vector, face box in the photo) for its largest
    face, or (None, None). Tries the photo as it is, then at 640 px, because
    YuNet misses a face in a very large or very small image.
    """
    img = decode_image(data)
    if img is None:
        return None, None
    # A phone photo can be 4000 px wide; detecting on it would hold the shared
    # lock (and pause recognition on every kiosk) far longer than needed.
    h, w = img.shape[:2]
    if max(h, w) > 1280:
        s = 1280 / max(h, w)
        img = cv2.resize(img, (0, 0), fx=s, fy=s)
    for attempt in (img, None):
        if attempt is None:
            h, w = img.shape[:2]
            s = 640 / max(h, w)
            attempt = cv2.resize(img, (0, 0), fx=s, fy=s)
        h, w = attempt.shape[:2]
        with _cv_lock:
            _detector.setInputSize((w, h))
            _, faces = _detector.detect(attempt)
        if faces is not None:
            faces = [f for f in faces if all(math.isfinite(float(v)) for v in f[:15])]
        if faces:
            best = max(faces, key=lambda f: f[2] * f[3])
            return embed(attempt, best), (attempt, best)
    return None, None


def thumbnail_jpeg(crop_source):
    """A small square picture of the registered face (for the admin), as JPEG bytes."""
    if not crop_source:
        return None
    img, face = crop_source
    x, y, w, h = [int(v) for v in face[:4]]
    pad = int(max(w, h) * 0.35)
    H, W = img.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (240, 240))
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else None


def _cosine(a, b):
    with _cv_lock:
        return float(_recognizer.match(a, b, cv2.FaceRecognizerSF_FR_COSINE))


def person_score(vector, vectors):
    """
    How well `vector` matches one person's registered angles. The first angle
    is the straight-on one (the scan starts with LOOK CENTER), and a kiosk
    only matches straight-on faces, so it counts for half; the best of all
    angles counts for the other half. The best angle alone let a stranger
    match on one lucky side view.
    """
    if not vectors:
        return -1.0
    scores = [_cosine(vector, t) for t in vectors]
    return 0.5 * scores[0] + 0.5 * max(scores)


def rank(vector, faces):
    """[(score, employee_id, name), ...] best first, one entry per person."""
    out = [(person_score(vector, vectors), emp_id, name) for emp_id, name, vectors in faces]
    out.sort(key=lambda x: x[0], reverse=True)
    return out


def best_match(vector, faces):
    """
    The registered person `vector` belongs to, among one company's faces only.
    `faces` is [(employee_id, name, [np.array, ...]), ...] (several angles each).
    Returns (employee_id, name, score, reason). reason is "matched", or why
    not: "unknown" (nobody close enough) or "unsure" (two people too close to
    tell apart).
    """
    ranked = rank(vector, faces)
    if not ranked:
        return None, None, 0.0, "unknown"
    best_score, best_id, best_name = ranked[0]
    if best_score < MATCH_THRESHOLD:
        return None, None, best_score, "unknown"
    if len(ranked) > 1 and best_score - ranked[1][0] < MATCH_MARGIN:
        return None, None, best_score, "unsure"
    return best_id, best_name, best_score, "matched"


# After the head turn, the person only has to be shown to be the SAME person
# again. That is a narrower question than "who is this?", so it does not need
# the margin over the next-best person, and a slightly lower score will do. It
# used to need a full match, and a face that scored just under the bar or close
# to a look-alike left the person waiting at "look straight" until some frame
# happened to match -- which looked like the kiosk waited for a blink (13 Oct).
CONFIRM_SLACK = float(os.getenv("LENS_CONFIRM_SLACK", "0.05"))


def is_same_person(vector, faces, emp_id):
    # True when emp_id is the closest registered person and close enough.
    ranked = rank(vector, faces)
    if not ranked:
        return False
    best_score, best_id, _ = ranked[0]
    return str(best_id) == str(emp_id) and best_score >= MATCH_THRESHOLD - CONFIRM_SLACK


def face_width(face_row, scale=1.0):
    """Face width in pixels of the frame the kiosk sent."""
    try:
        return float(face_row[2]) * scale
    except Exception:
        return 0.0


def pose_of(face_row):
    """
    Rough head pose from YuNet's landmarks, for the 360° registration scan.
    Landmarks: right eye, left eye, nose, mouth corners (image coordinates).
    """
    try:
        lex, ley, rex, rey, nx, ny = face_row[4], face_row[5], face_row[6], face_row[7], face_row[8], face_row[9]
        eye_dist = max(1.0, abs(rex - lex))
        ratio_x = (nx - min(lex, rex)) / eye_dist
        ratio_y = (ny - (ley + rey) / 2) / eye_dist
        yaw = "RIGHT" if ratio_x < 0.38 else ("LEFT" if ratio_x > 0.62 else "CENTER")
        pitch = "UP" if ratio_y < 0.28 else ("DOWN" if ratio_y > 0.58 else "CENTER")
        return {"yaw": yaw, "pitch": pitch, "rx": round(float(ratio_x), 2), "ry": round(float(ratio_y), 2)}
    except Exception:
        return {"yaw": "UNKNOWN", "pitch": "UNKNOWN"}
