"""One saved observation per locked UE frame; fail on gaps or inconsistent time."""
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


def json_write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")


def to_bgr8(pixels):
    """SPEAR native BGR(A). Float data uses display clipping, not HDR preservation."""
    a = np.asarray(pixels)
    if a.ndim != 3 or a.shape[2] not in (3, 4) or min(a.shape[:2]) < 1:
        raise ValueError(f"Unexpected camera shape: {a.shape}")
    a = a[:, :, :3]
    if not np.isfinite(a).all():
        raise ValueError("Camera returned NaN/Inf")
    if a.dtype == np.uint8:
        return np.ascontiguousarray(a)
    if np.issubdtype(a.dtype, np.floating):
        return np.ascontiguousarray(np.rint(np.clip(a, 0, 1) * 255).astype(np.uint8))
    raise ValueError(f"Unsupported camera dtype: {a.dtype}")


def vector3(values):
    values = [float(x) for x in values]
    if len(values) != 3 or not all(math.isfinite(x) for x in values):
        raise ValueError("Pose must contain 3 finite numbers")
    return values


def checked_action(action):
    out = {k: float(action[k]) for k in ("forward", "turn")}
    if not all(math.isfinite(v) and -1 <= v <= 1 for v in out.values()):
        raise ValueError("Action must be finite and in [-1, 1]")
    return out


class Recorder:
    def __init__(self, out_dir, fps=30):
        self.out_dir = Path(out_dir)
        # Refuse overwrite, including an incomplete previous recording.
        self.out_dir.mkdir(parents=True, exist_ok=False)
        (self.out_dir / "frames").mkdir()
        self.fps = fps
        self.expected_idx = 0
        self.first_time = None
        self.previous = None
        self._f = (self.out_dir / "manifest.jsonl").open("x", encoding="utf-8")
        self._decisions = (self.out_dir / "decisions.jsonl").open("x", encoding="utf-8")
        self._events = (self.out_dir / "llm_events.jsonl").open("x", encoding="utf-8")

    def decision(self, event):
        self._decisions.write(json.dumps(event, allow_nan=False) + "\n")
        self._decisions.flush()

    def event(self, event):
        self._events.write(json.dumps(event, allow_nan=False) + "\n")
        self._events.flush()

    def record(self, frame_idx, action, action_id, bgr, location, rotation,
               engine_frame, sim_time, camera_location, camera_rotation):
        if frame_idx != self.expected_idx:
            raise RuntimeError(f"Record gap: expected {self.expected_idx}, got {frame_idx}")
        if not math.isfinite(sim_time):
            raise RuntimeError("Non-finite engine time")
        action = checked_action(action)
        if self.previous is not None:
            p = self.previous
            if engine_frame != p["engine_frame"] + 1:
                raise RuntimeError(f"Engine frame gap: {p['engine_frame']} -> {engine_frame}")
            dt = sim_time - p["sim_time"]
            if abs(dt - 1 / self.fps) > 0.001:
                raise RuntimeError(f"Engine dt is {dt:.8f}; expected {1/self.fps:.8f}")
            if action_id not in (p["action_id"], p["action_id"] + 1):
                raise RuntimeError("Action ID jumped or went backwards")
            if action_id == p["action_id"] and action != p["action"]:
                raise RuntimeError("Held action changed without a new decision")
        if self.first_time is None:
            self.first_time = sim_time
        if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("Recorder requires uint8 BGR")
        rel = f"frames/{frame_idx}.png"
        path = self.out_dir / rel
        if not cv2.imwrite(str(path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise RuntimeError(f"Failed to save frame {frame_idx}")
        row = {
            "frame_idx": frame_idx, "t": sim_time - self.first_time,
            "sim_time": sim_time, "engine_frame": int(engine_frame),
            "action": action, "action_id": action_id,
            "location": vector3(location), "rotation": vector3(rotation),
            "camera_location": vector3(camera_location),
            "camera_rotation": vector3(camera_rotation), "image": rel,
            "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "image_shape": list(bgr.shape), "mean_pixel": float(bgr.mean()),
        }
        self._f.write(json.dumps(row, allow_nan=False) + "\n")
        self._f.flush()
        self.previous = row
        self.expected_idx += 1
        return row

    def close(self):
        for f in (self._f, self._decisions, self._events):
            f.close()
