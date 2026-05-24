"""Demo 2: heading-locked straight-turn-straight manoeuvre.

Workflow
--------
1. Enable motor.
2. Start yaw stabilization so the vehicle holds whatever heading it is given.
3. Drive forward for a set duration.
4. Command a 180-degree heading change via /cmd_pose.
5. Wait for the vehicle to settle on the new heading.
6. Drive forward again for a set duration.
7. Stop motor and disable stabilization on exit.

The 180-degree rotation is expressed as a unit quaternion rotating around the
Z-axis: q = (x=0, y=0, z=1, w=0).
"""

from __future__ import annotations

import math
import time
from pathlib import Path

from nauticore import init_vehicle
from nauticore.communication import MessageBuilder


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VEHICLE_PATH = REPO_ROOT / "examples" / "auv1.veh"


def main() -> None:
    # ---------------------------------------------------------------------
    # Step 0: edit these variables to match your setup.
    # ---------------------------------------------------------------------
    vehicle_path       = str(DEFAULT_VEHICLE_PATH)
    forward_speed      = 20.0          # m/s along X
    forward_duration_s = 4.0          # seconds to drive forward each leg
    turn_settle_s      = 3.0          # seconds to wait after sending the turn pose
    cmd_vel_topic      = "/teleop/cmd_vel"
    cmd_pose_topic     = "/cmd_pose"

    # ---------------------------------------------------------------------
    # Step 1: initialize Nauticore.
    # ---------------------------------------------------------------------
    if not Path(vehicle_path).exists():
        raise FileNotFoundError(
            f"Vehicle file not found: {vehicle_path}. "
            "Update vehicle_path at the top of main()."
        )

    handles  = init_vehicle(vehicle_path)
    frontend = handles.frontend

    # ---------------------------------------------------------------------
    # Step 2: enter the frontend context and arm the vehicle.
    # The context manager handles registration, listener/timer threads, and
    # clean shutdown (including motor-off) when the block exits.
    # ---------------------------------------------------------------------
    with frontend:
        print("Step 2: enabling motor...")
        frontend.startMotor()
        time.sleep(3)  # brief pause so the enable message reaches the bridge

        # -----------------------------------------------------------------
        # Step 3: activate yaw stabilization.
        # startStabilization() calls /control_mixer/set_position_override so
        # the vehicle will hold whichever heading is set via /cmd_pose.
        # -----------------------------------------------------------------
        print("Step 3: starting yaw stabilization...")
        frontend.startStabilization()
        
        for t in range(3):
            pose_msg, pose_type = MessageBuilder.pose_command(
                ori_x=0.0,
                ori_y=0.0,
                ori_z=1.0,   # sin(π/2) for 180° rotation around Z
                ori_w=0.0,   # cos(π/2)
            )
            frontend.publish(topic=cmd_pose_topic, type_str=pose_type, message_dict=pose_msg)
            print(".", end="", flush=True)
            time.sleep(1)
        # -----------------------------------------------------------------
        # Step 4: drive forward (first leg).
        # Publish a constant velocity command at forward_speed until
        # forward_duration_s has elapsed.
        # -----------------------------------------------------------------
        print(f"Step 4: driving forward for {forward_duration_s}s at {forward_speed} m/s...")
        t_end = time.time() + forward_duration_s
        while time.time() < t_end:
            msg, type_str = MessageBuilder.velocity_command(lin_x=forward_speed)
            frontend.publish(topic=cmd_vel_topic, type_str=type_str, message_dict=msg)
            time.sleep(0.2)

        # Stop translation before commanding a turn.
        msg, type_str = MessageBuilder.velocity_command()
        frontend.publish(topic=cmd_vel_topic, type_str=type_str, message_dict=msg)
        time.sleep(0.2)

        # -----------------------------------------------------------------
        # Step 5: command a 180-degree heading change.
        # A 180-degree rotation around the Z-axis is the unit quaternion
        #   q = (x=0, y=0, z=1, w=0)
        # which is exactly unit-length: norm = sqrt(0+0+1+0) = 1.
        # Only the orientation matters here; position components are zero
        # because the position controller derives absolute position separately.
        # -----------------------------------------------------------------
        print("Step 5: commanding 180-degree heading turn via /cmd_pose...")
        for t in range(3):
            pose_msg, pose_type = MessageBuilder.pose_command(
                ori_x=0.0,
                ori_y=0.0,
                ori_z=0.0,   # sin(π/2) for 180° rotation around Z
                ori_w=1.0,   # cos(π/2)
            )
            frontend.publish(topic=cmd_pose_topic, type_str=pose_type, message_dict=pose_msg)
            print(".", end="", flush=True)
            time.sleep(1)
        print(f"  Waiting {turn_settle_s}s for vehicle to settle on new heading...")
        time.sleep(turn_settle_s)

        # -----------------------------------------------------------------
        # Step 6: drive forward (second leg).
        # The vehicle is now facing the opposite direction, so lin_x > 0 will
        # push it in the new forward direction.
        # -----------------------------------------------------------------
        print(f"Step 6: driving forward again for {forward_duration_s}s...")
        t_end = time.time() + forward_duration_s
        while time.time() < t_end:
            msg, type_str = MessageBuilder.velocity_command(lin_x=forward_speed)
            frontend.publish(topic=cmd_vel_topic, type_str=type_str, message_dict=msg)
            time.sleep(0.1)

        # Send a zero-velocity command to stop cleanly before shutdown.
        msg, type_str = MessageBuilder.velocity_command()
        frontend.publish(topic=cmd_vel_topic, type_str=type_str, message_dict=msg)
        time.sleep(0.2)

        # -----------------------------------------------------------------
        # Step 7: disable stabilization and motor.
        # stopStabilization() calls /control_mixer/set_position_override with
        # data=False. stopMotor() is also called automatically by shutdown()
        # when the context exits, but calling it here makes the intent clear.
        # -----------------------------------------------------------------
        print("Step 7: stopping stabilization and motor...")
        frontend.stopStabilization()
        frontend.stopMotor()

    print("Demo 4 complete.")


if __name__ == "__main__":
    main()
