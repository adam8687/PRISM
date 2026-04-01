"""
collect_safe_demos.py
=====================
Collect 50 SAFE demonstrations via keyboard teleoperation.
Saves to safe_demos.hdf5 in robosuite's standard demo format.

Keyboard controls
-----------------
  W / S       →  +X / -X  (forward / back)
  A / D       →  +Y / -Y  (left / right)
  Q / E       →  +Z / -Z  (up / down)
  Arrow keys  →  end-effector rotation
  Space       →  toggle gripper  (open → close → open …)
  R           →  reset and discard current episode
  Enter       →  save current episode and start a new one
  Esc         →  quit (saves whatever has been collected so far)

Tips for SAFE demos
-------------------
  • Move slowly — tap keys rather than holding them
  • Hover above the block before descending
  • Centre the gripper over the block before closing
  • Lift smoothly straight up
"""

import time
import json
import numpy as np
import h5py
import robosuite as suite
from robosuite.devices import Keyboard
from robosuite.utils.input_utils import input2action

# ── configuration ─────────────────────────────────────────────────────────────
OUTPUT_FILE    = "safe_demos.hdf5"
TARGET_DEMOS   = 50
ENV_NAME       = "Lift"
ROBOT          = "Panda"
CONTROL_FREQ   = 20

CONTROLLER_CONFIG = {
    "type": "BASIC",
    "body_parts": {
        "right": {
            "type": "OSC_POSE",
            "input_max": 1, "input_min": -1,
            "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
            "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
            "kp": 300, "damping_ratio": 1,
            "impedance_mode": "fixed",
            "kp_limits": [0, 300], "damping_ratio_limits": [0, 10],
            "position_limits": None, "orientation_limits": None,
            "uncouple_pos_ori": True, "control_delta": True,
            "interpolation": None, "ramp_ratio": 0.2,
            "gripper": {
                "type": "GRIP",
                "input_max": 1, "input_min": -1,
                "output_max": 1, "output_min": -1,
            }
        }
    }
}
# ──────────────────────────────────────────────────────────────────────────────


def make_env():
    return suite.make(
        env_name=ENV_NAME,
        robots=ROBOT,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=CONTROL_FREQ,
        controller_configs=CONTROLLER_CONFIG,
    )


def collect_demos(output_file: str, target_demos: int, demo_type: str = "safe"):
    env    = make_env()
    device = Keyboard(env=env)
    env.viewer.add_keypress_callback(device.on_press)

    saved_demos = []
    demo_idx    = 0

    print(f"\n{'='*60}")
    print(f"  Collecting {target_demos} {demo_type.upper()} demonstrations")
    print(f"  Output → {output_file}")
    print(f"{'='*60}")
    print(__doc__)

    while demo_idx < target_demos:
        print(f"\n[Demo {demo_idx + 1}/{target_demos}]  Reset — get ready …")
        obs    = env.reset()
        device.start_control()

        episode_obs     = []
        episode_actions = []
        episode_rewards = []
        episode_dones   = []

        while True:
            # ── read device, build action ──────────────────────────────────
            action, grasp = input2action(
                device=device,
                robot=env.robots[0],
                active_arm="right",
                env_configuration=None,
            )

            if action is None:          # Esc pressed
                print("\nEsc → quitting.")
                _write_hdf5(output_file, saved_demos, demo_type)
                env.close()
                return

            # ── step ───────────────────────────────────────────────────────
            next_obs, reward, done, info = env.step(action)
            env.render()

            episode_obs.append({k: v.copy() if isinstance(v, np.ndarray) else v
                                 for k, v in obs.items()})
            episode_actions.append(np.array(action, dtype=np.float32))
            episode_rewards.append(float(reward))
            episode_dones.append(bool(done))

            obs = next_obs

            # ── Enter → save episode ───────────────────────────────────────
            if device.get_controller_state().get("enter_pressed", False):
                if len(episode_actions) < 5:
                    print("  Episode too short — discarded.")
                else:
                    saved_demos.append({
                        "obs":     episode_obs,
                        "actions": episode_actions,
                        "rewards": episode_rewards,
                        "dones":   episode_dones,
                    })
                    demo_idx += 1
                    print(f"  ✓ Saved ({len(episode_actions)} steps, "
                          f"reward={sum(episode_rewards):.3f})")
                break

            # ── R → discard and reset ──────────────────────────────────────
            if device.get_controller_state().get("r_pressed", False):
                print("  Reset — discarding episode.")
                break

            if done:
                saved_demos.append({
                    "obs":     episode_obs,
                    "actions": episode_actions,
                    "rewards": episode_rewards,
                    "dones":   episode_dones,
                })
                demo_idx += 1
                print(f"  ✓ Task done automatically ({len(episode_actions)} steps, "
                      f"reward={sum(episode_rewards):.3f})")
                break

    _write_hdf5(output_file, saved_demos, demo_type)
    env.close()
    print(f"\nDone — {len(saved_demos)} demos saved to {output_file}")


def _write_hdf5(path: str, demos: list, demo_type: str):
    """Write collected demos to an HDF5 file in robosuite demo format."""
    if not demos:
        print("No demos to write.")
        return

    with h5py.File(path, "w") as f:
        # top-level metadata
        meta = f.create_group("data")
        meta.attrs["env"]       = ENV_NAME
        meta.attrs["robot"]     = ROBOT
        meta.attrs["demo_type"] = demo_type
        meta.attrs["n_demos"]   = len(demos)
        meta.attrs["collected"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        meta.attrs["controller_config"] = json.dumps(CONTROLLER_CONFIG)

        total_steps = 0
        for i, demo in enumerate(demos):
            grp = meta.create_group(f"demo_{i}")
            grp.attrs["n_steps"] = len(demo["actions"])
            grp.attrs["reward"]  = float(sum(demo["rewards"]))

            grp.create_dataset("actions",
                               data=np.array(demo["actions"], dtype=np.float32))
            grp.create_dataset("rewards",
                               data=np.array(demo["rewards"], dtype=np.float32))
            grp.create_dataset("dones",
                               data=np.array(demo["dones"],   dtype=bool))

            obs_grp = grp.create_group("obs")
            if demo["obs"]:
                for key in demo["obs"][0]:
                    vals = [o[key] for o in demo["obs"]]
                    if isinstance(vals[0], np.ndarray):
                        obs_grp.create_dataset(
                            key,
                            data=np.array(vals, dtype=np.float32))
                    else:
                        obs_grp.create_dataset(
                            key,
                            data=np.array(vals))

            total_steps += len(demo["actions"])

        meta.attrs["total_steps"] = total_steps

    print(f"\n  HDF5 written → {path}")
    print(f"  {len(demos)} demos, {total_steps} total steps")


if __name__ == "__main__":
    collect_demos(OUTPUT_FILE, TARGET_DEMOS, demo_type="safe")