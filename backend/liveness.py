"""
Is the face in front of the kiosk a real person, or a photo held up to it?

Before a recognised face is sent as a punch, the kiosk asks the person to turn
their head a little to one side (left or right, chosen at random) and then
look straight again. A real head turning moves the nose relative to the eyes;
a printed photo or a phone screen held up and tilted does not: the face on it
is flat, so the nose stays in the same place between the eyes whichever way
the card is turned.

Tested on 12 Oct with a face picture held up as a flat card in 2,224 positions
(turned -55..+55 degrees, tilted forward/back, rotated -45..+45 degrees in its
own plane, nearer and further), each offered as the answer to both "turn left"
and "turn right": none passed with the limits below. On a flat card the nose
only appears to move when the landmarks break down at steep angles, and then
the eyes are squeezed together (the card is seen edge-on); a real head turned
enough to move the nose 0.15 keeps most of its eye spacing. Looser limits let
some through (0.10 / 0.72: 61 positions; 0.14 / 0.85: 1).

Retuned on 16 Oct: people waited 15-20 s and some were stuck, because only a
turn of about 9-21 degrees passed. Turn less and the nose has not moved enough;
turn properly (30-45 degrees) and the eyes look closer together, which the old
0.85 eye-spacing limit read as a card seen edge-on. Measured on a synthetic
head (the face wrapped on a cylinder and rotated, nose standing out): the
limits below accept about 9-36 degrees. A second sign was added that a flat
picture does not give: the nose also moves along the line between the mouth
corners (it stands out in front of the mouth), so the eye-spacing limit could
be loosened. Photo-card test (qa/lens/real-user/liveness_test.py), before ->
now: portrait 1, 6 -> 4 of 2,842 poses; portrait 2, 186 -> 183 of 2,174.
Synthetic head accepted: 9-21 -> 9-36 degrees (portrait 1), 3-21 -> 3-24
(portrait 2). The kiosk now also shows how far along the turn is, and turns
the arrow round when the person has turned too far.

What this does NOT stop: a video of the person turning their head the right
way, played on a screen. The random direction and the time limit make that
hard, not impossible.

State is per kiosk and in memory: a restart only means the person is asked
again.
"""
import math
import os
import random
import threading
import time

ENABLED = os.getenv("LENS_LIVENESS", "1") not in ("0", "false", "off", "")
TURN_DELTA = float(os.getenv("LENS_LIVENESS_TURN", "0.15"))  # how far the nose must move (0.5 = straight on)
MOUTH_TURN = float(os.getenv("LENS_LIVENESS_MOUTH_TURN", "0.15"))  # ... and along the mouth line (0 = off)
RX_LIMIT = float(os.getenv("LENS_LIVENESS_RX_LIMIT", "0.88"))  # beyond this the landmarks are not trustworthy
MIN_EYE_RATIO = float(os.getenv("LENS_LIVENESS_MIN_EYE_RATIO", "0.75"))  # eye spacing vs straight on
TIMEOUT = float(os.getenv("LENS_LIVENESS_SECONDS", "12"))

_lock = threading.Lock()
_challenges = {}


MAX_ROLL = float(os.getenv("LENS_LIVENESS_MAX_ROLL", "25"))  # degrees the eye line may tilt


def _along(px, py, x1, y1, x2, y2):
    """Where a point sits along the line from (x1, y1) to (x2, y2), left to right: 0 .. 1."""
    if x2 < x1:
        x1, y1, x2, y2 = x2, y2, x1, y1
    dx, dy = x2 - x1, y2 - y1
    return ((px - x1) * dx + (py - y1) * dy) / max(1.0, dx * dx + dy * dy)


def measures(row):
    """
    Turn (where the nose sits along the line between the eyes: 0 = at one eye,
    1 = at the other, 0.5 = straight on), eye spacing relative to face height,
    and how much the eye line is tilted. Measured ALONG the eye line, so
    rotating a photo flat in front of the camera (tilting it like a steering
    wheel) cannot fake a turn: that moves the points but not where the nose
    sits between the eyes.
    """
    try:
        x1, y1, x2, y2 = float(row[4]), float(row[5]), float(row[6]), float(row[7])
        nx, ny = float(row[8]), float(row[9])
        if x2 < x1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        dx, dy = x2 - x1, y2 - y1
        d2 = max(1.0, dx * dx + dy * dy)
        t = ((nx - x1) * dx + (ny - y1) * dy) / d2
        h = max(1.0, float(row[3]))
        m = {"rx": t, "eye": math.sqrt(d2) / h, "h": h, "roll": math.degrees(math.atan2(dy, dx)),
             "cx": float(row[0]) + float(row[2]) / 2, "cy": float(row[1]) + h / 2, "w": max(1.0, float(row[2]))}
        m["mx"] = _along(nx, ny, float(row[10]), float(row[11]), float(row[12]), float(row[13])) if len(row) >= 14 else None
        return m
    except Exception:
        return None


def turn_state(ch, row):
    """
    How the asked turn is going, for this frame: ("ok" | "more" | "too_far" |
    "other_way" | "lost", progress 0..1). Only "ok" passes. The kiosk shows
    the progress and the state as the moving head (no words), so the person
    sees how much further to turn, or that they turned too far.
    """
    m = measures(row)
    if not m:
        return "lost", 0.0
    move = m["rx"] - ch["base"]["rx"]
    if ch["direction"] == "right":
        move = -move
    progress = max(0.0, min(1.0, move / TURN_DELTA)) if TURN_DELTA > 0 else 1.0
    _note(ch, move, m)
    # the same face at about the same distance and place, not someone else stepping in
    if not (0.7 <= m["h"] / ch["base"]["h"] <= 1.4 and same_place(ch, m)):
        return "lost", progress
    if abs(m["roll"]) > MAX_ROLL:  # head (or card) tilted sideways, not turned
        return "too_far", progress
    if move < -TURN_DELTA / 2:
        return "other_way", 0.0
    # Turned so far that the landmarks give out (nose outside the eyes, or the
    # eyes squeezed together): a real head should come back a little; a card
    # seen edge-on looks exactly like this and must not pass.
    if not (1 - RX_LIMIT) <= m["rx"] <= RX_LIMIT or m["eye"] < MIN_EYE_RATIO * ch["base"]["eye"]:
        return "too_far", progress
    if move < TURN_DELTA:
        return "more", progress
    # A real nose stands out in front of the mouth as well as the eyes, so it
    # moves along the mouth line too. A flat picture's "turn" is landmark noise
    # and mostly does not.
    bmx, mx = ch["base"].get("mx"), m.get("mx")
    if MOUTH_TURN > 0 and bmx is not None and mx is not None:
        mmove = mx - bmx
        if ch["direction"] == "right":
            mmove = -mmove
        if mmove < MOUTH_TURN:
            return "more", progress
    return "ok", 1.0


def turned_enough(ch, row):
    """True when this face shows a real head turned the asked way."""
    return turn_state(ch, row)[0] == "ok"


def _note(ch, move, m):
    """Best turn seen during this check, for the log line (tuning from real faces)."""
    st = ch.setdefault("stats", {"frames": 0, "best": -1.0, "eye_at_best": None})
    st["frames"] += 1
    if move > st["best"]:
        st["best"] = move
        st["eye_at_best"] = m["eye"] / max(1e-6, ch["base"]["eye"])


def summary(ch, outcome, now=None):
    """One plain log line about a finished check."""
    st = ch.get("stats") or {}
    took = (now or time.time()) - ch.get("started", time.time())
    line = (f"[liveness] {outcome} emp=...{str(ch.get('emp_id', ''))[-6:]} dir={ch.get('direction')} "
            f"took={took:.1f}s frames={st.get('frames', 0)}")
    if st.get("eye_at_best") is not None:
        line += f" best_move={st['best']:.3f} eye_at_best={st['eye_at_best']:.2f}"
    return line


def same_place(ch, m):
    b = ch["base"]
    if "cx" not in b or "cx" not in m:
        return True
    return abs(m["cx"] - b["cx"]) <= 0.6 * b["w"] and abs(m["cy"] - b["cy"]) <= 0.6 * b["w"]


def tracked_row(ch, rows):
    """
    The face this check is about: the one nearest where the person stood when
    it started (not simply the biggest face, which may be someone who has just
    stepped up). None when nobody is there any more.
    """
    best, best_d = None, None
    b = ch["base"]
    for r in rows:
        m = measures(r)
        if not m or not same_place(ch, m):
            continue
        d = (m["cx"] - b.get("cx", m["cx"])) ** 2 + (m["cy"] - b.get("cy", m["cy"])) ** 2
        if best_d is None or d < best_d:
            best, best_d = r, d
    return best


def start(kiosk, emp_id, name, score, row, now=None):
    """Ask the person just recognised to turn their head. Returns the challenge."""
    ch = {
        "emp_id": str(emp_id), "name": name, "score": score,
        "direction": random.choice(("left", "right")),
        "base": measures(row) or {"rx": 0.5, "eye": 0.3, "h": 100.0},
        "stage": "turn", "started": now or time.time(),
    }
    with _lock:
        _challenges[str(kiosk)] = ch
    return ch


def current(kiosk, now=None):
    """(challenge, expired_challenge): the one in progress, or the one that just ran out of time."""
    now = now or time.time()
    with _lock:
        ch = _challenges.get(str(kiosk))
        if ch and now - ch["started"] > TIMEOUT + 30:
            # Left behind by a tablet that went away (tab closed) long ago: drop
            # it quietly. Reporting it as just expired would reset the person's
            # waiting time and ask them again for a punch already done elsewhere.
            _challenges.pop(str(kiosk), None)
            return None, None
        if ch and now - ch["started"] > TIMEOUT:
            _challenges.pop(str(kiosk), None)
            return None, ch
        # tidy checks of other tablets that went away
        for k in [k for k, c in _challenges.items() if now - c["started"] > TIMEOUT + 30]:
            _challenges.pop(k, None)
        return ch, None


def clear(kiosk):
    with _lock:
        _challenges.pop(str(kiosk), None)
