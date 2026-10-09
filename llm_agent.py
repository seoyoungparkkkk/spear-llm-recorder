"""Asynchronous vision policy. Each consumed result is tied to its observation."""
import base64
import json
import math
import os
import re
import threading
import time

import cv2

SYSTEM = (
    "You control a small rolling sphere robot in a 3D scene. "
    "You see its first-person image, world position, heading and recent positions. "
    'Reply ONLY with JSON {"forward": <float -1..1>, "turn": <float -1..1>}. '
    "forward>0 moves ahead; forward<0 reverses; turn>0 turns right; turn<0 turns left. "
    "Avoid nearby walls and obstacles and explore toward the goal. "
    "If recent positions barely change despite forward commands, use the image "
    "to choose a different maneuver, such as reversing and turning. "
    "A command is held for several frames until your next decision arrives."
)


def ask_llm(client, model, bgr, location, rotation, goal, last_action, history):
    small = cv2.resize(bgr, (384, 384))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise RuntimeError("Could not encode LLM observation")
    prompt = (
        f"Goal: {goal}\nPosition: {location}\n"
        f"Rotation (pitch,yaw,roll degrees): {rotation}\n"
        f"Previous action: {json.dumps(last_action)}\n"
        f"Recent sampled positions/actions: {json.dumps(history)}\nChoose the next action."
    )
    response = client.messages.create(
        model=model, max_tokens=200, system=SYSTEM,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
             "data": base64.b64encode(buf.tobytes()).decode("ascii")}},
            {"type": "text", "text": prompt}]}],
    )
    raw = "\n".join(getattr(b, "text", "") for b in response.content)
    match = re.search(r"\{[^{}]*\}", raw, re.S)
    if not match:
        raise ValueError("LLM did not return a JSON action")
    parsed = json.loads(match.group(0))
    action = {}
    for name in ("forward", "turn"):
        value = float(parsed[name])
        if not math.isfinite(value):
            raise ValueError("LLM returned a non-finite action")
        action[name] = max(-1.0, min(1.0, value))
    return {"action": action, "model": getattr(response, "model", model),
            "response_id": getattr(response, "id", None), "raw_response": raw,
            "prompt_text": prompt}


class LLMWorker:
    def __init__(self, goal, client=None, model=None):
        self.goal = goal
        self.model = model or os.environ.get("LLM_MODEL", "claude-opus-4-8")
        if client is None:
            import anthropic
            client = anthropic.Anthropic(timeout=30.0, max_retries=1)
        self.client = client
        self._lock = threading.Lock()
        self._busy = False
        self._result = None
        self._closed = False
        self._next_request = 1

    def submit(self, frame_idx, bgr, location, rotation, last_action, history):
        with self._lock:
            # A finished response must be consumed before another request is sent.
            if self._closed or self._busy or self._result is not None:
                return None
            self._busy = True
            request_id = self._next_request
            self._next_request += 1
        snapshot = bgr.copy()
        location, rotation = [float(v) for v in location], [float(v) for v in rotation]
        action = dict(last_action)
        history = json.loads(json.dumps(history))

        def run():
            started = time.monotonic()
            result = {"request_id": request_id, "observation_frame": frame_idx}
            try:
                result.update(ask_llm(self.client, self.model, snapshot, location,
                                     rotation, self.goal, action, history))
                result["status"] = "ok"
            except Exception as e:
                # No fake/fallback policy: retain the previous command on error.
                result.update(status="error", error_type=type(e).__name__, error=str(e)[:600])
            result["latency_seconds"] = time.monotonic() - started
            with self._lock:
                if not self._closed:
                    self._result = result
                self._busy = False

        threading.Thread(target=run, daemon=True).start()
        return request_id

    def poll(self):
        with self._lock:
            result, self._result = self._result, None
        return result

    def close(self):
        # An in-flight HTTP request may finish, but can no longer affect the episode.
        with self._lock:
            self._closed = True
