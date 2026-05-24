"""Demo 4: detect ArUco markers and display them in the camera preview."""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from nauticore import init_vehicle


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VEHICLE_PATH = REPO_ROOT / "examples" / "auv1.veh"


def main() -> None:
    # ---------------------------------------------------------------------
    # Step 0: edit these variables first.
    # ---------------------------------------------------------------------
    vehicle_path = str(DEFAULT_VEHICLE_PATH)
    preferred_camera_label = "cam1"

    # ---------------------------------------------------------------------
    # Step 1: initialize Nauticore.
    # ---------------------------------------------------------------------
    if not Path(vehicle_path).exists():
        raise FileNotFoundError(
            f"Vehicle file not found: {vehicle_path}. Update vehicle_path at the top of main()."
        )

    handles = init_vehicle(vehicle_path)

    # ---------------------------------------------------------------------
    # Step 2: choose a camera.
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

    # ---------------------------------------------------------------------
    # Step 3: create the ArUco detector.
    # ---------------------------------------------------------------------
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco support is unavailable. Install opencv-contrib-python.")

    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)

    # ---------------------------------------------------------------------
    # Step 4: initialize loop state.
    # ---------------------------------------------------------------------
    last_seq = -1

    try:
        print(f"Using vehicle file: {vehicle_path}")
        print(f"Displaying camera: {camera_label}")
        print("Workflow: read frame -> detect ArUco -> draw ids -> preview")
        print("Press q or Esc to exit.")

        while True:
            # -------------------------------------------------------------
            # Step 5: read the newest camera frame.
            # -------------------------------------------------------------
            frame, seq = camera.get_raw_frame()
            if frame is None or seq == last_seq:
                time.sleep(0.01)
                continue

            last_seq = seq

            # -------------------------------------------------------------
            # Step 6: detect markers in the current frame.
            # -------------------------------------------------------------
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            preview = frame.copy()

            detected_ids = []
            if ids is not None and len(ids) > 0:
                detected_ids = ids.flatten().tolist()
                aruco.drawDetectedMarkers(preview, corners, ids)

            # -------------------------------------------------------------
            # Step 7: draw a small overlay.
            # -------------------------------------------------------------
            status_text = f"{camera_label} seq={seq} ids={detected_ids or 'none'}"
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
            # Step 8: show the preview window.
            # -------------------------------------------------------------
            cv2.imshow("nauticore aruco preview", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        # -----------------------------------------------------------------
        # Step 9: clean up all resources.
        # -----------------------------------------------------------------
        for camera in handles.cameras.values():
            try:
                camera.release()
            except Exception:
                pass

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()