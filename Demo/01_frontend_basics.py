"""Demo 1: basic Nauticore frontend usage in one readable script."""

from __future__ import annotations

import time
from pathlib import Path
import scipy.spatial.transform
import pandas as pd

from nauticore import init_vehicle
from nauticore.communication import MessageBuilder, Frontend


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VEHICLE_PATH = REPO_ROOT / "examples" / "auv1.veh"


def main() -> None:
    # ---------------------------------------------------------------------
    # Step 0: edit this variable first.
    # The goal of this file is to teach the core frontend APIs without hiding
    # the setup behind command-line parsing or helper wrappers.
    # ---------------------------------------------------------------------
    vehicle_path = str(DEFAULT_VEHICLE_PATH)

    # ---------------------------------------------------------------------
    # Step 1: initialize Nauticore from the vehicle file.
    # init_vehicle() returns one object that includes the frontend plus any
    # configured cameras. This demo focuses on frontend usage only.
    # ---------------------------------------------------------------------
    if not Path(vehicle_path).exists():
        raise FileNotFoundError(
            f"Vehicle file not found: {vehicle_path}. Update vehicle_path at the top of main()."
        )

    handles = init_vehicle(vehicle_path)
    frontend = handles.frontend
    # ---------------------------------------------------------------------
    # Step 2: create a small shared state object.
    # The timer callbacks below need to remember counts and timestamps, so the
    # demo keeps that state in one plain dictionary.
    # ---------------------------------------------------------------------
    state = {
        "heartbeat_count": 0,
        "service_attempted": False,
        "start_time": time.time(),
    }

    # ---------------------------------------------------------------------
    # Step 3: register topic callbacks.
    # route_msg("/topic") decorates a function that should be called whenever
    # the frontend receives a message on that topic.
    # ---------------------------------------------------------------------
    @frontend.route_msg("/odom")
    def handle_odom(message):
        if state.get("odom_printed"):
            return
        state["odom_printed"] = True
        pos = message["pose"]["pose"]["position"]
        quat = message["pose"]["pose"]["orientation"]
        eular = scipy.spatial.transform.Rotation.from_quat([quat["x"], quat["y"], quat["z"], quat["w"]]).as_euler("xyz", degrees=True)
        df = pd.DataFrame({
            "x": [pos["x"]],
            "y": [pos["y"]],
            "z": [pos["z"]],
            "roll": [eular[0]],
            "pitch": [eular[1]],
            "yaw": [eular[2]],
        })
        print(df)

    # ---------------------------------------------------------------------
    # Step 4: register a timer that periodically publishes a string topic.
    # timer(1000) means the decorated function runs every 1000 milliseconds.
    # ---------------------------------------------------------------------
    @frontend.timer(1000)
    def publish_heartbeat():
        state["heartbeat_count"] += 1
        payload = {
            "data": f"frontend-demo heartbeat #{state['heartbeat_count']} @ {int(time.time())}",
        }
        frontend.publish(
            topic="/heartbeat",
            type_str="std_msgs/String",
            message_dict=payload,
        )
        print(f"Published /heartbeat: {payload['data']}")

    # ---------------------------------------------------------------------
    # Step 5: register a second timer with a different frequency.
    # This demonstrates that multiple timer callbacks can run side by side.
    # ---------------------------------------------------------------------
    @frontend.timer(2500)
    def publish_elapsed_time():
        elapsed_s = round(time.time() - state["start_time"], 2)
        frontend.publish(
            topic="/demo/elapsed",
            type_str="std_msgs/Float32",
            message_dict={"data": elapsed_s},
        )
        print(f"Published /demo/elapsed: {elapsed_s}s")

    # ---------------------------------------------------------------------
    # Step 6: register a timer that performs one service call.
    # The callback guards itself so the service is only called once.
    # ---------------------------------------------------------------------
    @frontend.timer(1500)
    def call_demo_service_once():
        if state["service_attempted"]:
            return
        state["service_attempted"] = True

        try:
            # call_service() blocks until a response arrives or the timeout is hit.
            response = frontend.call_service(
                service_name="/control_mixer/set_position_override",
                type_str="std_srvs/SetBool",
                request_dict={"data": True},
                timeout=3,
            )
            print(f"Service /control_mixer/set_position_override response: {response}")
        except Exception as exc:
            print(f"Service /control_mixer/set_position_override failed: {exc}")

    try:
        # -----------------------------------------------------------------
        # Step 7: start the frontend event loop.
        # frontend.run() performs registration, starts listener and timer
        # workers, then keeps the process alive until the user stops it.
        # -----------------------------------------------------------------
        print(f"Using vehicle file: {vehicle_path}")
        print("Registered routes: /odom, /cmd_vel")
        print("Publishing: /heartbeat, /demo/elapsed")
        print("Calling service once: /demo/ping")

        frontend.run()
    finally:
        # -----------------------------------------------------------------
        # Step 8: clean up all resources.
        # Release any cameras created by init_vehicle(), then close the
        # frontend transport so no threads or sockets are left behind.
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


if __name__ == "__main__":
    main()