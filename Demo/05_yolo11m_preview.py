"""Demo 5: run a trained YOLO11m model on a Nauticore camera stream."""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from nauticore import init_vehicle

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError(
        "Demo 4 requires ultralytics. Install it with `pip install ultralytics`."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VEHICLE_PATH = REPO_ROOT / "examples" / "auv1.veh"


def main() -> None:
    # ---------------------------------------------------------------------
    # Step 0: edit these variables first.
    # trained_model_path should point to your exported YOLO11m weights file,
    # for example best.pt after training.
    # ---------------------------------------------------------------------
    vehicle_path = str(DEFAULT_VEHICLE_PATH)
    preferred_camera_label = "cam1"
    trained_model_path = REPO_ROOT / "weights" / "best.pt"
    confidence_threshold = 0.35
    image_size = 640

    # ---------------------------------------------------------------------
    # Step 1: validate inputs and initialize Nauticore.
    # ---------------------------------------------------------------------
    if not Path(vehicle_path).exists():
        raise FileNotFoundError(
            f"Vehicle file not found: {vehicle_path}. Update vehicle_path at the top of main()."
        )
    if not trained_model_path.exists():
        raise FileNotFoundError(
            f"YOLO weights not found: {trained_model_path}. Update trained_model_path at the top of main()."
        )

    handles = init_vehicle(vehicle_path)

    # ---------------------------------------------------------------------
    # Step 2: choose a camera and load the model.
    # ---------------------------------------------------------------------
    cameras = handles.cameras
    if not cameras:
        raise RuntimeError("No cameras are available in the initialized vehicle handles.")

    if preferred_camera_label in cameras:
        camera_label = preferred_camera_label
        camera = cameras[preferred_camera_label]
    else:
        camera_label = next(iter(cameras))
        camera = cameras[camera_label]

    model = YOLO(str(trained_model_path))

    # ---------------------------------------------------------------------
    # Step 3: initialize loop state.
    # ---------------------------------------------------------------------
    last_seq = -1
    last_infer_ms = 0.0

    try:
        print(f"Using vehicle file: {vehicle_path}")
        print(f"Displaying camera: {camera_label}")
        print(f"Using YOLO weights: {trained_model_path}")
        print("Workflow: read frame -> run YOLO -> draw boxes -> preview")
        print("Press q or Esc to exit.")

        while True:
            # -------------------------------------------------------------
            # Step 4: read the newest camera frame.
            # -------------------------------------------------------------
            frame, seq = camera.get_raw_frame()
            if frame is None or seq == last_seq:
                time.sleep(0.01)
                continue

            last_seq = seq

            # -------------------------------------------------------------
            # Step 5: run YOLO on the frame.
            # -------------------------------------------------------------
            started = time.perf_counter()
            results = model.predict(
                source=frame,
                conf=confidence_threshold,
                imgsz=image_size,
                verbose=False,
            )
            last_infer_ms = (time.perf_counter() - started) * 1000.0

            preview = frame.copy()
            detection_count = 0

            if results:
                result = results[0]
                preview = result.plot()
                if result.boxes is not None:
                    detection_count = len(result.boxes)

            # -------------------------------------------------------------
            # Step 6: draw an overlay with runtime info.
            # -------------------------------------------------------------
            status_text = (
                f"{camera_label} seq={seq} detections={detection_count} "
                f"infer={last_infer_ms:.1f}ms"
            )
            cv2.putText(
                preview,
                status_text,
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            # -------------------------------------------------------------
            # Step 7: show the preview window.
            # -------------------------------------------------------------
            cv2.imshow("nauticore yolo11m preview", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        # -----------------------------------------------------------------
        # Step 8: clean up all resources.
        # -----------------------------------------------------------------
        for camera in handles.cameras.values():
            try:
                camera.release()
            except Exception:
                pass

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()