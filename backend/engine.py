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
import os

import cv2
import numpy as np

MODEL_VERSION = "sface_2021dec"
# SFace cosine threshold. 0.363 is OpenCV's published value; 0.42 is a little
# stricter, chosen because a wrong match punches the wrong person.
MATCH_THRESHOLD = float(os.getenv("LENS_MATCH_THRESHOLD", "0.42"))

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
    _detector.setInputSize((w, h))
    _, faces = _detector.detect(img)
    return img, ([] if faces is None else list(faces)), scale


def embed(img, face_row):
    """128-value SFace vector for one detected face."""
    aligned = _recognizer.alignCrop(img, face_row)
    return _recognizer.feature(aligned)[0]


def encode_photo(data: bytes):
    """
    One registration photo -> (vector, face box in the photo) for its largest
    face, or (None, None). Tries the photo as it is, then at 640 px, because
    YuNet misses a face in a very large or very small image.
    """
    img = decode_image(data)
    if img is None:
        return None, None
    for attempt in (img, None):
        if attempt is None:
            h, w = img.shape[:2]
            s = 640 / max(h, w)
            attempt = cv2.resize(img, (0, 0), fx=s, fy=s)
        h, w = attempt.shape[:2]
        _detector.setInputSize((w, h))
        _, faces = _detector.detect(attempt)
        if faces is not None and len(faces) > 0:
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


def best_match(vector, faces):
    """
    The closest registered face to `vector`, among one company's faces only.
    `faces` is [(employee_id, name, [np.array, ...]), ...] (several angles each).
    Returns (employee_id, name, score) above the threshold, else (None, None, score).
    """
    best_id, best_name, best_score = None, None, -1.0
    for emp_id, name, vectors in faces:
        for target in vectors:
            score = float(_recognizer.match(vector, target, cv2.FaceRecognizerSF_FR_COSINE))
            if score > best_score:
                best_id, best_name, best_score = emp_id, name, score
    if best_score >= MATCH_THRESHOLD:
        return best_id, best_name, best_score
    return None, None, best_score


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
