"""VL client: screen -> Qwen3-VL server -> structured game state JSON.

Talks to a local llama.cpp llama-server (OpenAI-compatible) running
Qwen3-VL with an mmproj, e.g. via Desktop/launch-vl.bat on port 8080.

Usage:
    from vl_client import get_vl_state
    state = get_vl_state()          # cached, refreshes at most every MIN_INTERVAL_S
    state = get_vl_state(force=True)

Returns a dict:
    {
        "available": bool,          # did the server answer successfully
        "age": float,               # seconds since last successful analysis
        "scene": str,               # combat|menu|dialogue|death|victory|unknown
        "health_pct": int | None,
        "threat_dir": str,          # left|right|up|down|none|unknown
        "enemies_visible": bool | None,
        "next_action": str,         # one of config action keys, or none
        "raw": str                  # model's raw reply
    }
"""

import base64
import io
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections import deque

from PIL import Image

_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")

# ---------------------------------------------------------------------------
# Configuration (override with env vars)
# ---------------------------------------------------------------------------
VL_SERVER = os.environ.get("VL_SERVER", "http://127.0.0.1:8080")
MIN_INTERVAL_S = float(os.environ.get("VL_MIN_INTERVAL_S", "2.5"))
ERROR_BACKOFF_S = float(os.environ.get("VL_ERROR_BACKOFF_S", "5"))
MAX_WIDTH = int(os.environ.get("VL_MAX_WIDTH", "1024"))
JPEG_QUALITY = int(os.environ.get("VL_JPEG_QUALITY", "70"))
MAX_TOKENS = int(os.environ.get("VL_MAX_TOKENS", "160"))
TIMEOUT_S = float(os.environ.get("VL_TIMEOUT_S", "120"))

VALID_SCENES = {"combat", "menu", "dialogue", "death", "victory", "unknown"}
VALID_DIRS = {"left", "right", "up", "down", "none", "unknown"}
VALID_ACTIONS = {"W", "A", "S", "D", "SPACE", "SHIFT", "Q", "E", "F", "ALT", "TAB", "none"}

PROMPT = (
    "You are the vision module of a bot playing the game Hades. "
    "Analyze this screenshot and reply with ONLY a single line of minified JSON "
    "matching the required schema.\n"
    "CRITICAL GROUNDING RULES - never guess:\n"
    "- health_bar_visible: true ONLY if you can actually see the player's red "
    "health bar UI element (in Hades it sits at the BOTTOM-LEFT of the screen). "
    "If it is not clearly visible, set it false.\n"
    "- health_pct: estimate ONLY when health_bar_visible is true; otherwise use 0.\n"
    "- enemies_visible: true only if enemy characters are visibly present.\n"
    "- threat_dir: direction of the nearest visible enemy or incoming attack; "
    "'none' if no enemies.\n"
    "- next_action: single most useful key press right now "
    "(SPACE=attack, SHIFT=dash); in menus/dialogues pick a key that advances them."
)

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "scene": {"type": "string",
                  "enum": ["combat", "menu", "dialogue", "death", "victory", "unknown"]},
        "health_bar_visible": {"type": "boolean"},
        "health_pct": {"type": "integer"},
        "threat_dir": {"type": "string",
                       "enum": ["left", "right", "up", "down", "none", "unknown"]},
        "enemies_visible": {"type": "boolean"},
        "next_action": {"type": "string",
                        "enum": ["W", "A", "S", "D", "SPACE", "SHIFT",
                                 "Q", "E", "F", "ALT", "TAB", "none"]},
    },
    "required": ["scene", "health_bar_visible", "health_pct",
                 "threat_dir", "enemies_visible", "next_action"],
}

_EMPTY = {
    "available": False,
    "age": None,
    "scene": "unknown",
    "health_bar_visible": False,
    "health_pct": None,
    "threat_dir": "unknown",
    "enemies_visible": None,
    "next_action": "none",
    "raw": "",
}

_cache: dict = {"state": dict(_EMPTY), "ts": 0.0}
_history: deque = deque(maxlen=3)   # last N successful reads, for smoothing


def _mode(values):
    """Most common value; ties broken by most recent."""
    if not values:
        return None
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts.values())
    for v in reversed(values):
        if counts[v] == best:
            return v
    return values[-1]


def _smoothed():
    """Majority vote over recent reads; single outlier claims get outvoted."""
    if not _history:
        return dict(_EMPTY)
    scenes = [s["scene"] for s in _history]
    bars = [bool(s["health_bar_visible"]) for s in _history]
    enemies = [s["enemies_visible"] for s in _history
               if isinstance(s["enemies_visible"], bool)]
    threats = [s["threat_dir"] for s in _history]
    # Health % only trusted when the majority of recent reads saw the bar
    pcts = [s["health_pct"] for s in _history
            if s["health_bar_visible"] and s["health_pct"] is not None]
    return {
        "available": True,
        "age": round(time.time() - _cache["ts"], 2),
        "scene": _mode(scenes),
        "health_bar_visible": sum(bars) > len(bars) / 2,
        "health_pct": sorted(pcts)[len(pcts) // 2] if pcts else None,
        "threat_dir": _mode(threats),
        "enemies_visible": (_mode(enemies) if enemies else None),
        "next_action": _history[-1]["next_action"],
        "raw": _history[-1]["raw"],
        "samples": len(_history),
    }


def capture_frame():
    """Capture and downscale the game view to a base64 JPEG string."""
    try:
        from game_capture import capture_game_window, capture_screen_fallback
        shot = capture_game_window() or capture_screen_fallback()
    except ImportError:
        import pyautogui
        shot = pyautogui.screenshot()
    if shot.width > MAX_WIDTH:
        ratio = MAX_WIDTH / shot.width
        shot = shot.resize((MAX_WIDTH, int(shot.height * ratio)), _LANCZOS)
    buf = io.BytesIO()
    shot.convert("RGB").save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _extract_json(text):
    """Pull the first JSON object out of a model reply."""
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _clamp(state, data):
    scene = str(data.get("scene", "unknown")).lower().strip()
    state["scene"] = scene if scene in VALID_SCENES else "unknown"

    bar = data.get("health_bar_visible")
    if isinstance(bar, bool):
        state["health_bar_visible"] = bar
    elif isinstance(bar, str):
        state["health_bar_visible"] = bar.lower() in ("true", "yes", "1")

    hp = data.get("health_pct")
    # Accept a number ONLY when the model also claims the element is visible
    if state["health_bar_visible"] and isinstance(hp, (int, float)) and 0 <= hp <= 100:
        state["health_pct"] = int(hp)

    d = str(data.get("threat_dir", "unknown")).lower().strip()
    state["threat_dir"] = d if d in VALID_DIRS else "unknown"

    ev = data.get("enemies_visible")
    if isinstance(ev, bool):
        state["enemies_visible"] = ev
    elif isinstance(ev, str):
        state["enemies_visible"] = ev.lower() in ("true", "yes", "1")

    act = str(data.get("next_action", "none")).upper().strip()
    state["next_action"] = act if act in VALID_ACTIONS else "none"


def _fetch_once():
    """Capture the screen, query the server once, update the cache."""
    try:
        b64 = capture_frame()
    except Exception as e:
        result = dict(_EMPTY, raw=f"capture failed: {e}")
        _cache.update(state=result, ts=time.time())
        return result

    payload = json.dumps({
        "model": "qwen3-vl",
        "max_tokens": MAX_TOKENS,
        "temperature": 0.1,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "game_state", "strict": True,
                            "schema": JSON_SCHEMA},
        },
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{VL_SERVER}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        raw = body["choices"][0]["message"]["content"]
    except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as e:
        result = dict(_EMPTY, raw=f"server error: {e}")
        _cache.update(state=result, ts=time.time())
        return result

    result = dict(_EMPTY)
    data = _extract_json(raw)
    if data:
        _clamp(result, data)
        result["available"] = True
    result["raw"] = raw
    result["age"] = 0.0
    if result["available"]:
        _history.append(dict(result))
    _cache.update(state=dict(_smoothed()), ts=time.time())
    return result


def _refresher():
    """Daemon loop: keep the cache fresh without blocking callers."""
    while True:
        try:
            result = _fetch_once()
            # On failure back off briefly so a dead server isn't hammered
            time.sleep(MIN_INTERVAL_S if result.get("available") else ERROR_BACKOFF_S)
        except Exception:
            time.sleep(ERROR_BACKOFF_S)


_refresher_thread = None


def _ensure_refresher():
    global _refresher_thread
    if _refresher_thread is None or not _refresher_thread.is_alive():
        _refresher_thread = threading.Thread(target=_refresher, daemon=True)
        _refresher_thread.start()


def get_vl_state(force=False):
    """Return latest game-state analysis instantly from the background cache.

    Never blocks (after the first call starts the refresher). Use force=True
    for a synchronous refresh (CLI/testing only).
    """
    if force:
        _fetch_once()
        return _smoothed()

    _ensure_refresher()
    result = dict(_cache["state"])
    age = time.time() - _cache["ts"] if _cache["ts"] > 0 else None
    result["age"] = round(age, 2) if age is not None else None
    return result


def is_server_up():
    """Cheap liveness check against the VL server."""
    try:
        with urllib.request.urlopen(f"{VL_SERVER}/props", timeout=3) as resp:
            props = json.loads(resp.read().decode("utf-8"))
            return bool(props.get("modalities", {}).get("vision"))
    except Exception:
        return False


if __name__ == "__main__":
    print(f"server: {VL_SERVER}  vision={is_server_up()}")
    print(json.dumps(get_vl_state(force=True), indent=2))
