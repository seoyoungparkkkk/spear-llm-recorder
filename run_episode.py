"""LLM-driven SPEAR sphere agent, explicit single stepping and checked frame recording."""
import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import socket
import sys
import time
import traceback

FPS = 30
FORCE_SCALE = 3000.0
YAW_STEP_DEG = 2.0
HERE = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(HERE / "user_config.yaml"))
    p.add_argument("--episode", default=None, help="New folder name; existing folders are never overwritten")
    p.add_argument("--frames", type=int, default=450)
    p.add_argument("--llm-every", type=int, default=15)
    p.add_argument("--frame-delay", type=float, default=0.0)
    p.add_argument("--preview", action="store_true", help="Show the actual agent camera + recorder counter")
    p.add_argument("--goal", default=None)
    args = p.parse_args()
    if args.frames < 300 or args.llm_every < 1 or not 0 <= args.frame_delay <= 1:
        p.error("Use frames >= 300, llm-every >= 1 and frame-delay between 0 and 1")
    if args.episode is None:
        args.episode = datetime.now().strftime("episode_%Y%m%d_%H%M%S_%f")
    if Path(args.episode).name != args.episode or args.episode in (".", ".."):
        p.error("episode must be a folder name, not a path")
    return args


def main():
    args = parse_args()
    import cv2
    import numpy as np
    import spear
    from dotenv import load_dotenv
    from llm_agent import LLMWorker
    from recorder import Recorder, json_write, to_bgr8
    from validate import validate_bundle, make_preview

    load_dotenv(HERE / ".env", override=False)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is missing. Use your existing .env beside run_episode.py.")
    goal = args.goal or os.environ.get("EPISODE_GOAL") or (
        "Explore the room toward a distinctive visible object. Avoid walls and obstacles. "
        "If blocked, find another direction using the camera and recent positions.")
    config = spear.get_config(user_config_files=[str(Path(args.config).expanduser().resolve())])
    config.defrost()
    init = config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE
    init.OVERRIDE_BENCHMARKING = True
    init.BENCHMARKING = True
    init.OVERRIDE_FIXED_DELTA_TIME = True
    init.FIXED_DELTA_TIME = 1.0 / FPS
    config.SPEAR.INSTANCE.CLIENT_INTERNAL_TIMEOUT_SECONDS = 60.0
    config.SPEAR.INSTANCE.INITIALIZE_CLIENT_MAX_TIME_SECONDS = 300.0
    config.SPEAR.INSTANCE.REQUEST_TERMINATE_UNREAL_INSTANCE_MAX_TIME_SECONDS = 30.0
    config.freeze()
    if config.SPEAR.LAUNCH_MODE not in ("editor", "game"):
        raise ValueError("This runner launches its own instance: use LAUNCH_MODE editor or game.")
    with socket.socket() as s:
        s.settimeout(0.5)
        if s.connect_ex(("127.0.0.1", config.SP_SERVICES.RPC_SERVICE.RPC_SERVER_PORT)) == 0:
            raise RuntimeError("SPEAR port is busy. Quit the previous Unreal app with Cmd+Q, then retry.")
    out = HERE / "recordings" / args.episode
    worker = LLMWorker(goal)
    rec = Recorder(out, fps=FPS)
    info = {"status": "running", "requested_frames": args.frames, "recorded_frames": 0,
            "fps": FPS, "single_step": True, "llm_every": args.llm_every,
            "model_requested": worker.model, "goal": goal, "termination": "fixed_frame_budget",
            "agent": "BP_SphereAgent", "pose_convention": "sphere center + navigation root heading",
            "pose_units": "Unreal centimeters; pitch/yaw/roll degrees",
            "observation_timing": "post-action, end of same engine frame; first captured frame has t=0",
            "warmup_frames_excluded": 30, "force_scale": FORCE_SCALE, "yaw_step_deg": YAW_STEP_DEG,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "goal_success_evaluated": False, "approved_scene_agent_verified": False,
            "pixel_conversion": "uint8 unchanged; floating BGR clipped to [0,1], multiplied by 255",
            "sources_sha256": {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                for name in ("run_episode.py", "recorder.py", "llm_agent.py", "validate.py")}}
    json_write(out / "run_info.json", info)
    (out / "resolved_config.yaml").write_text(config.dump(), encoding="utf-8")
    print(f"[episode] output: {out}", flush=True)
    print(f"[episode] goal: {goal}", flush=True)
    instance = None
    completed = False
    try:
        spear.configure_system(config=config)
        instance = spear.Instance(config=config)
        in_editor = (config.SPEAR.LAUNCH_MODE == "editor"
                     and config.SPEAR.INSTANCE.EDITOR_LAUNCH_MODE == "editor")
        if in_editor:
            editor = instance.get_editor()
            with instance.begin_frame():
                subsystem = editor.unreal_service.get_editor_subsystem(uclass="ULevelEditorSubsystem")
                subsystem.EditorRequestBeginPlay()
            with instance.end_frame():
                pass
        game = instance.get_game()
        with instance.begin_frame():
            game.console_service.set_cvar(name="r.Streaming.FramesForFullUpdate", value=0)
            game.console_service.set_cvar(name="r.Streaming.FullyLoadUsedTextures", value=1)
            gameplay = game.get_unreal_object(uclass="UGameplayStatics")
            system = game.get_unreal_object(uclass="UKismetSystemLibrary")
            info["scene_name"] = gameplay.GetCurrentLevelName(bRemovePrefixString=True)
            uclass = game.unreal_service.load_class(uclass="AActor", name="/SpContent/Blueprints/BP_SphereAgent.BP_SphereAgent_C")
            agent = game.unreal_service.spawn_actor(uclass=uclass, location={"X": -10., "Y": 280., "Z": 150.})
            root = game.unreal_service.get_component_by_name(actor=agent, component_name="DefaultSceneRoot", uclass="USceneComponent")
            camera = game.unreal_service.get_component_by_name(actor=agent, component_name="DefaultSceneRoot.final_tone_curve_hdr_", uclass="USpSceneCaptureComponent2D")
            body = game.unreal_service.get_component_by_path(actor=agent, component_path="DefaultSceneRoot.sphere_", uclass="USceneComponent")
            camera.Width = 512
            camera.Height = 512
            camera.Initialize()
            camera.initialize_sp_funcs()
        with instance.end_frame():
            pass
        game.async_loading_service.wait_for_engine_idle()
        # Let the sphere settle before the recorded episode starts.
        for _ in range(30):
            with instance.begin_frame():
                gameplay.SetGamePaused(bPaused=False)
            with instance.end_frame(single_step=True):
                gameplay.SetGamePaused(bPaused=True)

        action, action_id = {"forward": 0.0, "turn": 0.0}, 0
        history = deque(maxlen=8)
        for idx in range(args.frames):
            with instance.begin_frame():
                before_frame = int(system.GetFrameCount())
                gameplay.SetGamePaused(bPaused=False)
                root.K2_AddRelativeRotation(DeltaRotation={"Pitch": 0., "Yaw": action["turn"]*YAW_STEP_DEG, "Roll": 0.})
                rotation = root.K2_GetComponentRotation()
                matrix = spear.math.to_numpy_matrix_from_spear_rotator(spear_rotator=rotation)
                force = matrix @ np.array([action["forward"]*FORCE_SCALE, 0., 0.])
                body.AddForce(Force=spear.math.to_spear_vector_from_numpy_array(numpy_array=force))
            # single_step holds at the next begin-frame while Python saves/encodes data.
            with instance.end_frame(single_step=True):
                pixels = camera.read_pixels()["arrays"]["data"].copy()
                loc = spear.math.to_numpy_array_from_spear_vector(body.K2_GetComponentLocation())
                rot = spear.math.to_numpy_array_from_spear_rotator(root.K2_GetComponentRotation())
                cam_loc = spear.math.to_numpy_array_from_spear_vector(camera.K2_GetComponentLocation())
                cam_rot = spear.math.to_numpy_array_from_spear_rotator(camera.K2_GetComponentRotation())
                engine_frame = int(system.GetFrameCount())
                sim_time = float(gameplay.GetTimeSeconds())
                gameplay.SetGamePaused(bPaused=True)
            if before_frame != engine_frame:
                raise RuntimeError(f"Action/observation frame mismatch: {before_frame} != {engine_frame}")
            bgr = to_bgr8(pixels)
            if idx == 0:
                info["native_camera_dtype"] = str(pixels.dtype)
                info["native_camera_range_first_frame"] = [float(pixels.min()), float(pixels.max())]
            row = rec.record(idx, action, action_id, bgr, loc, rot,
                             engine_frame, sim_time, cam_loc, cam_rot)
            if args.preview:
                preview = bgr.copy()
                cv2.rectangle(preview, (0, 0), (512, 54), (0, 0, 0), -1)
                cv2.putText(preview, f"SAVED {idx+1}/{args.frames} | UE frame {engine_frame}", (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
                cv2.putText(preview, f"action {action_id}: forward={action['forward']:.2f} turn={action['turn']:.2f}",
                            (8, 43), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
                cv2.imshow("SPEAR agent camera + recorder", preview)
                cv2.waitKey(1)
            # No response is marked 'applied' unless a next recorded frame exists.
            if idx < args.frames - 1:
                result = worker.poll()
                if result is not None:
                    if result["status"] == "ok":
                        action_id += 1
                        action = result["action"]
                        observed = result["observation_frame"]
                        result.update(action_id=action_id, apply_frame=idx+1,
                            observation_image=f"frames/{observed}.png",
                            observation_image_sha256=hashlib.sha256((out/"frames"/f"{observed}.png").read_bytes()).hexdigest())
                        rec.decision(result)
                        print(f"[LLM] action {action_id} starts at frame {idx+1}: {action}", flush=True)
                    else:
                        rec.event(result)
                        print(f"[LLM] error {result['error_type']}: {result['error']}", flush=True)
                if idx % args.llm_every == 0:
                    history.append({"frame": idx, "location": row["location"], "action": row["action"]})
                    request = worker.submit(idx, bgr, loc, rot, row["action"], list(history))
                    if request is not None:
                        rec.event({"status": "submitted", "request_id": request, "observation_frame": idx})
            if idx % 30 == 0 or idx == args.frames-1:
                print(f"[record] {idx+1}/{args.frames} UE={engine_frame} t={row['t']:.3f} "
                      f"action_id={row['action_id']} loc={np.round(loc,1).tolist()}", flush=True)
            if args.frame_delay:
                time.sleep(args.frame_delay)
        # Consume the final single-step boundary, then release normal engine execution.
        with instance.begin_frame():
            gameplay.SetGamePaused(bPaused=True)
        with instance.end_frame():
            camera.terminate_sp_funcs()
            camera.Terminate()
            game.unreal_service.destroy_actor(actor=agent)
        if in_editor:
            game = None
            with instance.begin_frame():
                subsystem.EditorRequestEndPlay()
            with instance.end_frame():
                pass
        completed = True
        info["status"] = "complete"
    except BaseException:
        info["status"] = "failed"
        (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"[episode] FAILED; details: {out / 'error.txt'}", flush=True)
        raise
    finally:
        worker.close()
        rec.close()
        info["recorded_frames"] = rec.expected_idx
        json_write(out / "run_info.json", info)
        if args.preview:
            cv2.destroyAllWindows()
        if instance is not None:
            try:
                instance.close(force=not completed)
            except Exception as e:
                print(f"[cleanup] {type(e).__name__}: {e}", flush=True)
                try:
                    instance.close(force=True)
                except Exception:
                    print("[cleanup] Close this run's Unreal app manually if it remains open.")
    result = validate_bundle(out)
    json_write(out / "validation.json", result)
    make_preview(out)
    for name, ok in result["checks"].items():
        print("PASS" if ok else "FAIL", name)
    for warning in result["warnings"]:
        print("NOTE", warning)
    print("RECORDING PASS" if result["recording_pass"] else "RECORDING FAIL", flush=True)
    print(f"[episode] results: {out}", flush=True)
    if sys.platform == "darwin":
        import subprocess
        subprocess.run(["open", str(out)], check=False)
    return 0 if result["recording_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
