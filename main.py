import robosuite as suite
import numpy as np

controller_config = {
    "type": "BASIC",
    "body_parts": {
        "right": {
            "type": "OSC_POSE",
            "input_max": 1,
            "input_min": -1,
            "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
            "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
            "kp": 300,
            "damping_ratio": 1,
            "impedance_mode": "fixed",
            "kp_limits": [0, 300],
            "damping_ratio_limits": [0, 10],
            "position_limits": None,
            "orientation_limits": None,
            "uncouple_pos_ori": True,
            "control_delta": True,
            "interpolation": None,
            "ramp_ratio": 0.2,
            "gripper": {
                "type": "GRIP",
                "input_max": 1,
                "input_min": -1,
                "output_max": 1,
                "output_min": -1,
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
    control_freq=20,
    controller_configs=controller_config
)

obs = env.reset()

def get_action(obs, phase):

    ee_pos  = obs["robot0_eef_pos"]
    block_pos = obs["cube_pos"]
    delta   = block_pos - ee_pos
    move_xy = np.clip(delta[:2], -0.05, 0.05)
    no_rot  = [0, 0, 0]

    if phase == "approach":
        target_z = block_pos[2] + 0.15        # hover above block
        dz = np.clip(target_z - ee_pos[2], -0.05, 0.05)
        return [*move_xy, dz, *no_rot, -1]    # gripper open

    elif phase == "descend":
        target_z = block_pos[2] - 0.02        # slightly below center
        dz = np.clip(target_z - ee_pos[2], -0.05, 0.05)
        return [*move_xy, dz, *no_rot, -1]    # gripper open

    elif phase == "grasp":
        return [0, 0, 0, *no_rot, 1]          # close gripper, don't move

    elif phase == "lift":
        return [0, 0, 0.05, *no_rot, 1]       # move up, keep closed

def reached(obs, target_pos, threshold=0.015):
    ee_pos = obs["robot0_eef_pos"]
    return np.linalg.norm(ee_pos - target_pos) < threshold

def run_phase(phase, max_steps=300):
    """Run a phase until the goal condition is met or max_steps is hit."""
    block_pos = obs["cube_pos"]

    if phase == "approach":
        goal = np.array([block_pos[0], block_pos[1], block_pos[2] + 0.15])
    elif phase == "descend":
        goal = np.array([block_pos[0], block_pos[1], block_pos[2] - 0.02])
    else:
        goal = None   # grasp/lift just run for fixed steps

    for step in range(max_steps):
        action = get_action(obs, phase)
        o, reward, done, info = env.step(action)
        env.render()

        # update obs in outer scope
        globals()["obs"] = o

        if goal is not None and reached(o, goal):
            print(f"  ✓ {phase} reached target in {step+1} steps")
            return reward, done

        if done:
            return reward, done

    print(f"  ✗ {phase} hit max steps ({max_steps})")
    return reward, done

phases = ["approach", "descend", "grasp", "lift"]

reward, done = 0, False
for phase in phases:
    print(f"\n>>> Phase: {phase}")
    reward, done = run_phase(phase, max_steps=300)
    if done:
        print(f"Task complete! reward={reward:.3f}")
        break

print(f"\nFinal reward: {reward:.3f}")
env.close()