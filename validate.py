"""Local verification of saved data. This is NOT the organizer's official grader."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from recorder import checked_action, json_write, vector3


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_bundle(directory, demo=None):
    root = Path(directory).resolve()
    checks, details, warnings = {}, {}, []

    def check(name, test):
        try:
            checks[name] = bool(test())
        except Exception as e:
            checks[name] = False
            details[name] = f"{type(e).__name__}: {e}"

    try:
        rows = load_jsonl(root / "manifest.jsonl")
        info = json.loads((root / "run_info.json").read_text())
        decisions = load_jsonl(root / "decisions.jsonl")
    except Exception as e:
        return {"recording_pass": False, "submission_checks_pass": False,
                "checks": {"read_bundle": False}, "details": {"read_bundle": str(e)},
                "warnings": ["Not an official grading result."]}
    n = len(rows)
    fps = float(info.get("fps", 30))
    check("completed_run", lambda: info.get("status") == "complete"
          and n == info["requested_frames"] == info["recorded_frames"])
    check("contiguous_300plus_records", lambda: n >= 300
          and all(type(r["frame_idx"]) is int for r in rows)
          and [r["frame_idx"] for r in rows] == list(range(n)))

    def valid_fields():
        for r in rows:
            vector3(r["location"])
            vector3(r["rotation"])
            vector3(r["camera_location"])
            vector3(r["camera_rotation"])
            checked_action(r["action"])
            if not all(math.isfinite(r[k]) for k in ("t", "sim_time")):
                return False
            if type(r["engine_frame"]) is not int or type(r["action_id"]) is not int:
                return False
            if r["action_id"] < 0 or not isinstance(r["image"], str):
                return False
        return n > 0

    check("required_fields_and_finite_values", valid_fields)
    check("engine_frames_contiguous", lambda: n > 1 and all(
        b["engine_frame"] == a["engine_frame"] + 1 for a, b in zip(rows, rows[1:])))

    def valid_time():
        if n < 2 or not math.isfinite(fps) or fps <= 0:
            return False
        first = rows[0]["sim_time"]
        if any(abs(r["t"] - (r["sim_time"] - first)) > 1e-7 for r in rows):
            return False
        return all(0 < b["t"] - a["t"] and
                   abs((b["sim_time"] - a["sim_time"]) - 1/fps) <= 0.001
                   for a, b in zip(rows, rows[1:]))

    check("measured_engine_time_matches_fps", valid_time)

    def valid_images():
        expected = {f"{i}.png" for i in range(n)}
        actual = {p.name for p in (root / "frames").iterdir()
                  if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")}
        if expected != actual:
            return False
        for i, r in enumerate(rows):
            if r["image"] != f"frames/{i}.png":
                return False
            path = root / r["image"]
            decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if decoded is None or list(decoded.shape) != r["image_shape"]:
                return False
            if hashlib.sha256(path.read_bytes()).hexdigest() != r["image_sha256"]:
                return False
        return n > 0

    check("images_decode_count_and_hash", valid_images)

    def valid_actions():
        if not rows or rows[0]["action_id"] != 0:
            return False
        applied = []
        for previous, current in zip(rows, rows[1:]):
            if current["action_id"] == previous["action_id"]:
                if current["action"] != previous["action"]:
                    return False
            elif current["action_id"] == previous["action_id"] + 1:
                applied.append(current)
            else:
                return False
        if len(applied) != len(decisions) or not decisions or n <= len(decisions):
            return False
        for i, (r, d) in enumerate(zip(applied, decisions), start=1):
            if (d["action_id"] != i or d["apply_frame"] != r["frame_idx"]
                    or d["action"] != r["action"] or d["status"] != "ok"
                    or not 0 <= d["observation_frame"] < d["apply_frame"]):
                return False
            observation = rows[d["observation_frame"]]
            if d["observation_image_sha256"] != observation["image_sha256"]:
                return False
        return True

    check("held_actions_and_llm_decision_alignment", valid_actions)
    if rows:
        try:
            locations = np.asarray([r["location"] for r in rows])
            distances = np.linalg.norm(np.diff(locations[:, :2], axis=0), axis=1)
            movement = {"xy_path_length_unreal_cm": float(distances.sum()),
                        "xy_net_displacement_unreal_cm": float(np.linalg.norm(
                            locations[-1, :2] - locations[0, :2]))}
            if distances.sum() < 1:
                warnings.append("Almost no horizontal motion; inspect navigation and images.")
            if len({r["image_sha256"] for r in rows}) < 2:
                warnings.append("All saved images are identical; inspect the capture.")
            if all(r["mean_pixel"] < 2 for r in rows):
                warnings.append("All images are very dark; inspect the camera before submission.")
        except (KeyError, ValueError, TypeError):
            movement = {}
    else:
        movement = {}
    recording_pass = all(checks.values())
    duration = None
    if demo is not None:
        def valid_demo():
            nonlocal duration
            cap = cv2.VideoCapture(str(demo))
            try:
                rate = cap.get(cv2.CAP_PROP_FPS)
                count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                ok, _ = cap.read()
                duration = count/rate if rate > 0 else 0
                return ok and math.isfinite(duration) and 30 <= duration <= 90
            finally:
                cap.release()
        check("demo_30_to_90_seconds", valid_demo)
    else:
        warnings.append("Demo not checked. A separate 30-90 second live screen demo is required.")
    warnings.append("Navigation success, approved scene/agent and demo content need human review.")
    return {"recording_pass": recording_pass,
            "submission_checks_pass": recording_pass and checks.get("demo_30_to_90_seconds", False),
            "frames": n, "applied_llm_decisions": len(decisions),
            "checks": checks, "details": details, "movement": movement,
            "demo_duration_seconds": duration, "warnings": warnings,
            "official_grader": False}


def make_preview(directory):
    root = Path(directory)
    rows = load_jsonl(root / "manifest.jsonl")
    if not rows:
        return
    tiles = []
    for i in np.linspace(0, len(rows)-1, 12, dtype=int):
        image = cv2.imread(str(root / rows[i]["image"]))
        if image is None:
            return
        tile = cv2.resize(image, (256, 256))
        cv2.rectangle(tile, (0, 0), (256, 27), (0, 0, 0), -1)
        cv2.putText(tile, f"frame {i} / action {rows[i]['action_id']}", (6, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(tile)
    preview = np.vstack([np.hstack(tiles[k:k+4]) for k in (0, 4, 8)])
    if not cv2.imwrite(str(root / "preview.jpg"), preview):
        raise RuntimeError("Could not write preview.jpg")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory")
    p.add_argument("demo", nargs="?", help="Optional live demo file; required for submission check")
    args = p.parse_args()
    result = validate_bundle(args.directory, args.demo)
    for name, ok in result["checks"].items():
        print("PASS" if ok else "FAIL", name)
    for warning in result["warnings"]:
        print("NOTE", warning)
    json_write(Path(args.directory) / "validation.json", result)
    print("RECORDING PASS" if result["recording_pass"] else "RECORDING FAIL")
    if args.demo:
        print("LOCAL SUBMISSION CHECKS PASS" if result["submission_checks_pass"]
              else "LOCAL SUBMISSION CHECKS FAIL")
    return 0 if (result["submission_checks_pass"] if args.demo else result["recording_pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
