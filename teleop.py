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

RENDER_EVERY = 2  # render every N steps to reduce GPU load

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

while True:
    obs = env.reset()
    device.start_control()

    step = 0
    last_state = None

    while True:
        # ── read raw device state ──────────────────────────────────────────
        state = device.get_controller_state()

        # Uncomment the next line if nothing moves — shows you the real keys:
        # print(state)

        dpos  = state.get("dpos",      np.zeros(3))
        drot  = state.get("drotation", np.zeros(3))
        grasp = state.get("grasp",     0)
        reset = state.get("reset",     False)

        # quit if Esc was pressed (some versions set dpos to None)
        if dpos is None or state.get("quit", False):
            env.close()
            raise SystemExit

        if reset:
            print("  Resetting …")
            break

        # skip physics step when robot is idle and state hasn't changed
        def state_changed(a, b):
            if a is None:
                return True
            for k in a:
                va, vb = a[k], b[k]
                if isinstance(va, np.ndarray):
                    if not np.array_equal(va, vb):
                        return True
                elif va != vb:
                    return True
            return False

        is_idle = np.allclose(dpos, 0) and np.allclose(drot, 0)
        if not is_idle or state_changed(last_state, state):
            action = np.concatenate([dpos, drot, [float(grasp)]])
            obs, reward, done, info = env.step(action)
            last_state = state

            if reward > 0:
                print(f"  ✓ block lifted!  reward={reward:.3f}")

            if done:
                print("  Episode done — resetting")
                break

        step += 1
        if step % RENDER_EVERY == 0:
            env.render()