"""Demo 6: detect ArUco 4x4 tags and steer /teleop/cmd_vel from marker error."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from nauticore import init_vehicle
from nauticore.communication import MessageBuilder
import imutils


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VEHICLE_PATH = REPO_ROOT / "examples" / "auv1.veh"


def _seed_tracking_points(gray: np.ndarray, marker_corners: np.ndarray) -> np.ndarray:
    polygon = np.round(marker_corners).astype(np.int32)
    mask = np.zeros_like(gray)
    cv2.fillConvexPoly(mask, polygon, 255)

    points = [marker_corners.reshape(-1, 1, 2).astype(np.float32)]
    feature_points = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=40,
        qualityLevel=0.01,
        minDistance=4,
        mask=mask,
        blockSize=5,
    )
    if feature_points is not None and len(feature_points) > 0:
        points.append(feature_points.astype(np.float32))

    return np.concatenate(points, axis=0)


def _track_marker(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    previous_points: np.ndarray,
    previous_corners: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        previous_points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )

    if next_points is None or status is None:
        return None

    good_old = previous_points[status.flatten() == 1]
    good_new = next_points[status.flatten() == 1]
    if len(good_old) < 4 or len(good_new) < 4:
        return None

    transform, _ = cv2.estimateAffinePartial2D(good_old, good_new)
    if transform is None:
        return None

    transformed_corners = cv2.transform(
        previous_corners.reshape(-1, 1, 2),
        transform,
    ).reshape(-1, 2)
    transformed_points = cv2.transform(
        previous_points.reshape(-1, 1, 2),
        transform,
    ).reshape(-1, 1, 2)
    return transformed_points.astype(np.float32), transformed_corners.astype(np.float32)


def main() -> None:
    # ---------------------------------------------------------------------
    # Step 0: edit these three variables first.
    # A learner should be able to understand this demo by reading top to
    # bottom and changing only these values, without chasing helper methods.
    # ---------------------------------------------------------------------
    vehicle_path = str(DEFAULT_VEHICLE_PATH)
    preferred_camera_label = "cam1"
    publish_interval_s = 0.05   # seconds between velocity publishes
    target_id = 24              # ArUco ID to track and centre
    ang_z_gain = -2           # proportional gain: ang_z = -gain * normalised_error
    z_gain = 3               # proportional gain: z = -gain * normalised_error
    tracker_max_gap_frames = 8  # how many consecutive frames may rely on tracking only

    # ---------------------------------------------------------------------
    # Step 1: initialize Nauticore.
    # init_vehicle() is the main entry point used in these demos. It reads the
    # vehicle file and returns one object containing the frontend connection
    # plus any configured camera handles.
    # ---------------------------------------------------------------------
    if not Path(vehicle_path).exists():
        raise FileNotFoundError(
            f"Vehicle file not found: {vehicle_path}. Update vehicle_path at the top of main()."
        )

    handles = init_vehicle(vehicle_path)
    frontend = handles.frontend

    # ---------------------------------------------------------------------
    # Step 2: choose the camera.
    # The code first tries the requested label. If that label is missing, it
    # falls back to the first available camera so the demo still runs on other
    # vehicle configurations.
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
    # This demo uses OpenCV's 4x4_50 dictionary. Keeping the detector setup in
    # main() makes it obvious which OpenCV API calls are needed.
    # ---------------------------------------------------------------------
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco support is unavailable. Install opencv-contrib-python.")

    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)

    # ---------------------------------------------------------------------
    # Step 4: initialize loop state.
    # last_seq prevents duplicate processing of the same frame. The publish
    # timestamp limits how frequently cmd_vel is sent. motor_enabled tracks
    # whether the one-time enable topic has already been published.
    # ---------------------------------------------------------------------
    last_seq = -1
    last_publish_time = 0.0
    motor_enabled = False
    prev_gray: np.ndarray | None = None
    tracked_points: np.ndarray | None = None
    tracked_corners: np.ndarray | None = None
    tracker_gap_frames = tracker_max_gap_frames + 1
    fps_start_time = time.time()
    fps_frame_count = 0
    fps_display = 0.0

    try:
        # -----------------------------------------------------------------
        # Step 5: enter the frontend context.
        # Frontend now supports with-style usage, so entering the context starts
        # the connection and leaving it shuts everything down automatically.
        # -----------------------------------------------------------------
        with frontend:
            print(f"Using vehicle file: {vehicle_path}")
            print(f"Using camera: {camera_label}")
            print(f"Publish interval: {publish_interval_s}s")
            print("Workflow: read frame -> detect ArUco -> fallback tracker -> enable motor -> publish /teleop/cmd_vel")
            print("Press q or Esc to exit.")
            if not motor_enabled:
                frontend.publish(
                    topic="/eth/motor_enable",
                    type_str="std_msgs/Int32",
                    message_dict={"data": 1},
                )
                print("Published /eth/motor_enable -> {'data': 1}")
                motor_enabled = True
            while True:
                # ---------------------------------------------------------
                # Step 6: read the newest camera frame.
                # get_raw_frame() returns both the current image and a sequence
                # number. If the sequence did not change, there is no new frame yet.
                # ---------------------------------------------------------
                frame, seq = camera.get_raw_frame()
                if frame is None or seq == last_seq:
                    time.sleep(0.01)
                    continue

                last_seq = seq
                fps_frame_count += 1
                fps_elapsed = time.time() - fps_start_time
                if fps_elapsed >= 1.0:
                    fps_display = fps_frame_count / fps_elapsed
                    fps_frame_count = 0
                    fps_start_time = time.time()
                # frame= cv2.rotate(frame, cv2.ROTATE_180)
                # ---------------------------------------------------------
                # Step 7: detect ArUco markers.
                # The frame is converted to grayscale first because that is the most
                # common input format for OpenCV ArUco detection.
                # ---------------------------------------------------------
                preview = frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = detector.detectMarkers(gray)

                detected_ids = []
                target_corners = None
                target_source = "hold"
                now = time.time()

                if ids is not None and len(ids) > 0:
                    detected_ids = ids.flatten().tolist()

                    # -----------------------------------------------------
                    # Step 8: visualize the detected markers.
                    # Drawing marker boxes makes it easy to confirm that OpenCV is
                    # seeing the same tags that the operator sees on screen.
                    # -----------------------------------------------------
                    aruco.drawDetectedMarkers(preview, corners, ids)

                    # Find the corners of target_id if it is visible.
                    for i, mid in enumerate(ids.flatten()):
                        if mid == target_id:
                            target_corners = corners[i][0].astype(np.float32)
                            tracked_corners = target_corners.copy()
                            tracked_points = _seed_tracking_points(gray, tracked_corners)
                            tracker_gap_frames = 0
                            target_source = "aruco"
                            break

                if (
                    target_corners is None
                    and prev_gray is not None
                    and tracked_points is not None
                    and tracked_corners is not None
                    and tracker_gap_frames < tracker_max_gap_frames
                ):
                    tracked_result = _track_marker(prev_gray, gray, tracked_points, tracked_corners)
                    if tracked_result is not None:
                        tracked_points, tracked_corners = tracked_result
                        target_corners = tracked_corners.copy()
                        tracker_gap_frames += 1
                        target_source = "tracker"
                        cv2.polylines(
                            preview,
                            [np.round(target_corners).astype(np.int32)],
                            isClosed=True,
                            color=(0, 165, 255),
                            thickness=2,
                        )
                    else:
                        tracked_points = None
                        tracked_corners = None
                        tracker_gap_frames = tracker_max_gap_frames + 1

                if target_corners is None:
                    tracker_gap_frames = min(tracker_gap_frames + 1, tracker_max_gap_frames + 1)

                # ---------------------------------------------------------
                # Step 9: visual servo — rotate to keep target_id centred.
                # The horizontal error between the marker centre and the image
                # centre drives ang_z proportionally. Positive error (marker is
                # to the right) -> negative ang_z (rotate right / CW in ROS).
                # When target_id is not visible ang_z = 0 (hold position).
                # ---------------------------------------------------------
                if now - last_publish_time >= publish_interval_s:
                    if target_corners is not None:

                        frame_cx  = frame.shape[1] / 2.0
                        marker_cx = target_corners[:, 0].mean()
                        errorx  = marker_cx - frame_cx
                        ang_z  = ang_z_gain * errorx / frame_cx  # 去负号：如果右边(+)->向右转(+)
                        # 角度范围：[-1, 1]
                        frame_cy  = frame.shape[0] / 2.0
                        marker_cy = target_corners[:, 1].mean()
                        errory  = marker_cy - frame_cy
                        zu  = z_gain * errory / frame_cy
                        print(
                            f"id={target_id} source={target_source} marker_cx={marker_cx:.1f} "
                            f"errorx={errorx:.1f}px ang_z={ang_z:.3f} lin_z={zu:.3f}"
                        )
                    else:
                        ang_z = 0.0
                        zu = 0.0
                        # print(f"id={target_id} not visible — holding (detected={detected_ids})")

                    twist, type_str = MessageBuilder.velocity_command(ang_z=ang_z, lin_z=zu)
                    frontend.publish(
                        topic="/teleop/cmd_vel",
                        type_str=type_str,
                        message_dict=twist,
                    )
                    last_publish_time = now

                # ---------------------------------------------------------
                # Step 10: draw a small overlay for the live preview.
                # This is just local UI so the learner can see which camera is being
                # used, which frame is displayed, and which marker IDs were found.
                # ---------------------------------------------------------
                status_text = (
                    f"{camera_label} fps={fps_display:.1f} seq={seq} ids={detected_ids or 'none'} source={target_source} "
                    f"gap={tracker_gap_frames if target_source == 'tracker' else 0}"
                )
                preview = imutils.resize(preview, width=1280)    
                cv2.putText(
                    preview,
                    status_text,
                    (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                # ---------------------------------------------------------
                # Step 11: show the preview window.
                # The window refreshes continuously until the user presses q or Esc.
                # ---------------------------------------------------------
                preview = imutils.resize(preview, width=1280)
                cv2.imshow("nauticore aruco demo", preview)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    frontend.publish(
                        topic="/eth/motor_enable",
                        type_str="std_msgs/Int32",
                        message_dict={"data": 0},
                    )
                    
                    print("Published /eth/motor_enable -> {'data': 0}")
                    break

                prev_gray = gray
    finally:
        # -----------------------------------------------------------------
        # Step 12: clean up all resources.
        # Release every camera created by init_vehicle(), then destroy OpenCV
        # windows. The frontend context manager already handles shutdown.
        # -----------------------------------------------------------------
        for camera in handles.cameras.values():
            try:
                camera.release()
            except Exception:
                pass

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()