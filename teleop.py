"""
teleop.py  —  keyboard control for robosuite Lift
==================================================
W / S          forward / back      (+X / -X)
A / D          left / right        (+Y / -Y)
Q / E          up / down           (+Z / -Z)
Arrow keys     rotate end-effector
Space          toggle gripper
R              reset environment
Esc            quit
"""

import copy
import numpy as np
import robosuite as suite
from robosuite.devices import Keyboard

CONTROLLER_CONFIG = {
    "type": "BASIC",
    "body_parts": {
        "right": {
            "type": "OSC_POSE",
            "input_max": 1, "input_min": -1,
            "output_max": [0.2, 0.2, 0.2, 1.0, 1.0, 1.0],
            "output_min": [-0.2, -0.2, -0.2, -1.0, -1.0, -1.0],
            "kp": 300, "damping_ratio": 1,
            "impedance_mode": "fixed",
            "kp_limits": [0, 300], "damping_ratio_limits": [0, 10],
            "position_limits": None, "orientation_limits": None,
            "uncouple_pos_ori": True, "control_delta": True,
            "interpolation": None, "ramp_ratio": 0.8,
            "gripper": {
                "type": "GRIP",
                "input_max": 1, "input_min": -1,
                "output_max": 1, "output_min": -1,
            }
        }
    }
}

env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    camera_names=[],
    control_freq=50,
    controller_configs=CONTROLLER_CONFIG,
)

device = Keyboard(env=env)
env.viewer.add_keypress_callback(device.on_press)

print(__doc__)


def state_snapshot(state: dict) -> dict:
    """Deep-copy a controller state dict so comparisons work on held keys."""
    snap = {}
    for k, v in state.items():
        snap[k] = v.copy() if isinstance(v, np.ndarray) else v
    return snap


def state_changed(prev: dict | None, curr: dict) -> bool:
    if prev is None:
        return True
    for k in curr:
        va, vb = prev.get(k), curr[k]
        if isinstance(vb, np.ndarray):
            if not np.array_equal(va, vb):
                return True
        elif va != vb:
            return True
    return False


while True:
    obs = env.reset()
    device.start_control()
    last_snap = None

    while True:
        state = device.get_controller_state()

        dpos  = state.get("dpos",      np.zeros(3))
        drot  = state.get("drotation", np.zeros(3))
        grasp = state.get("grasp",     0)
        reset = state.get("reset",     False)

        # Quit if Esc was pressed (some builds set dpos to None)
        if dpos is None or state.get("quit", False):
            env.close()
            raise SystemExit

        if reset:
            print("  Resetting …")
            break

        is_idle = np.allclose(dpos, 0) and np.allclose(drot, 0)

        if not is_idle or state_changed(last_snap, state):
            # FIX 1: snapshot state *before* using it so held-key comparison
            #         is always against an independent copy, not the live dict.
            last_snap = state_snapshot(state)

            # FIX 2: map grasp (0 / 1) → gripper command (-1 = open, +1 = close).
            #         Without this the gripper stalls at 0 (half-open) and
            #         can never generate enough force to hold the block.
            gripper_cmd = 1.0 if grasp else -1.0

            action = np.concatenate([dpos, drot, [gripper_cmd]])
            obs, reward, done, info = env.step(action)

            if reward > 0:
                print(f"  ✓ block lifted!  reward={reward:.3f}")

            if done:
                print("  Episode done — resetting")
                break

        # FIX 3: render every step so the display keeps up with physics.
        #         Skipping frames makes inputs feel laggy even when the
        #         underlying control loop is running at full speed.
        env.render()