"""Demo 3: basic Nauticore camera preview in one readable script."""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from nauticore import init_vehicle


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VEHICLE_PATH = REPO_ROOT / "examples" / "auv1.veh"


def main() -> None:
    # ---------------------------------------------------------------------
    # Step 0: edit these two variables first.
    # This demo is meant to show the minimum Nauticore camera workflow without
    # pushing the reader through argument parsing or helper wrappers.
    # ---------------------------------------------------------------------
    vehicle_path = str(DEFAULT_VEHICLE_PATH)
    preferred_camera_label = "cam1"

    # ---------------------------------------------------------------------
    # Step 1: initialize Nauticore.
    # init_vehicle() creates one object that includes the frontend connection
    # and every configured camera stream.
    # ---------------------------------------------------------------------
    if not Path(vehicle_path).exists():
        raise FileNotFoundError(
            f"Vehicle file not found: {vehicle_path}. Update vehicle_path at the top of main()."
        )

    handles = init_vehicle(vehicle_path)
    frontend = handles.frontend

    # ---------------------------------------------------------------------
    # Step 2: choose a camera.
    # The demo tries the requested label first and falls back to the first
    # available camera so the file stays usable on different vehicles.
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
    # Step 3: initialize loop state.
    # last_seq is used to skip duplicate frames when no fresh image has been
    # delivered by the camera stream yet.
    # ---------------------------------------------------------------------
    last_seq = -1

    try:

        while True:
            # -------------------------------------------------------------
            # Step 4: read the newest frame from the selected camera.
            # get_raw_frame() returns both the image and a sequence number.
            # If the sequence did not change, the demo waits briefly and tries
            # again instead of redrawing the same frame.
            # -------------------------------------------------------------
            frame, seq = camera.get_raw_frame()
        

            # -------------------------------------------------------------
            # Step 5: create a local preview image.
            # The frame is copied so the overlay text can be drawn without
            # mutating the original image object returned by the camera.
            # -------------------------------------------------------------
            

            # -------------------------------------------------------------
            # Step 6: draw a small overlay.
            # This makes it obvious which camera stream is active and which
            # frame sequence number is currently on screen.
            # -------------------------------------------------------------


            # -------------------------------------------------------------
            # Step 7: show the preview window.
            # The loop keeps refreshing until the user presses q or Esc.
            # -------------------------------------------------------------



    finally:
        # -----------------------------------------------------------------
        # Step 8: clean up all resources.
        # Release every camera created by init_vehicle(), close the frontend
        # transport, and then destroy OpenCV windows.
        # -----------------------------------------------------------------
        for camera in handles.cameras.values():
            try:
                camera.release()
            except Exception:
                pass

        try:
            if getattr(frontend, "running", False):
                frontend.shutdown()
            else:
                frontend.sock.close()
        except Exception:
            pass

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()