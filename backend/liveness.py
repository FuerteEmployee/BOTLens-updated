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
RX_LIMIT = float(os.getenv("LENS_LIVENESS_RX_LIMIT", "0.85"))  # beyond this the landmarks are not trustworthy
MIN_EYE_RATIO = float(os.getenv("LENS_LIVENESS_MIN_EYE_RATIO", "0.85"))  # eye spacing vs straight on
TIMEOUT = float(os.getenv("LENS_LIVENESS_SECONDS", "10"))

_lock = threading.Lock()
_challenges = {}


MAX_ROLL = float(os.getenv("LENS_LIVENESS_MAX_ROLL", "25"))  # degrees the eye line may tilt


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
        return {"rx": t, "eye": math.sqrt(d2) / h, "h": h, "roll": math.degrees(math.atan2(dy, dx))}
    except Exception:
        return None


def turned_enough(ch, row):
    """True when this face shows a real head turned the asked way."""
    m = measures(row)
    if not m:
        return False
    move = m["rx"] - ch["base"]["rx"]
    if ch["direction"] == "right":
        move = -move
    if move < TURN_DELTA:
        return False
    if not (1 - RX_LIMIT) <= m["rx"] <= RX_LIMIT:  # nose outside the eyes: landmarks broke, as on a tilted card
        return False
    if m["eye"] < MIN_EYE_RATIO * ch["base"]["eye"]:  # eyes squeezed like a card seen edge-on
        return False
    if abs(m["roll"]) > MAX_ROLL:  # head (or card) tilted sideways, not turned
        return False
    # the same face at about the same distance, not someone else stepping in
    return 0.7 <= m["h"] / ch["base"]["h"] <= 1.4


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
        if ch and now - ch["started"] > TIMEOUT:
            _challenges.pop(str(kiosk), None)
            return None, ch
        return ch, None


def clear(kiosk):
    with _lock:
        _challenges.pop(str(kiosk), None)
