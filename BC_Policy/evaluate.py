"""
evaluate.py — Roll out a trained BC policy in a robosuite environment.

Usage
-----
  python evaluate.py                                   # uses best_policy.pt
  python evaluate.py --ckpt checkpoints/best_policy.pt --n_rollouts 50
  python evaluate.py --render                          # open a viewer window
  python evaluate.py --record_video rollout.mp4        # save video (requires cv2)

The script prints per-episode success/return and aggregate statistics.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from model import build_policy


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate BC policy in robosuite")
    p.add_argument("--ckpt",          default="checkpoints/best_policy.pt")
    p.add_argument("--n_rollouts",    type=int,   default=20)
    p.add_argument("--max_ep_len",    type=int,   default=500)
    p.add_argument("--render",        action="store_true")
    p.add_argument("--record_video",  default=None,
                   help="Path to save an .mp4 of the first rollout")
    p.add_argument("--camera",        default="agentview")
    p.add_argument("--video_height",  type=int, default=512)
    p.add_argument("--video_width",   type=int, default=512)
    p.add_argument("--seed",          type=int,   default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Env builder
# ---------------------------------------------------------------------------

def build_env(env_name: str, env_args: dict, obs_keys: list[str], render: bool):
    """Reconstruct the robosuite environment from the saved env_args."""
    import robosuite as suite
    from robosuite.wrappers import GymWrapper

    # Ensure the obs keys we care about are active
    env_args = dict(env_args)          # shallow copy — don't mutate checkpoint
    env_args.setdefault("robots", "Panda")
    env_args.setdefault("has_renderer",          render)
    env_args.setdefault("has_offscreen_renderer", True)
    env_args.setdefault("use_camera_obs",         False)   # state-only by default
    env_args.setdefault("reward_shaping",         False)
    env_args.setdefault("control_freq",           20)

    env = suite.make(env_name, **env_args)
    return env


def extract_obs_vector(obs_dict: dict, obs_keys: list[str]) -> np.ndarray:
    parts = []
    # Define a mapping for known naming mismatches
    mapping = {
        "egg_pos": "Can_pos",       # Map 'egg' to 'Can'
        "egg_quat": "Can_quat",
        "eef_pos": "robot0_eef_pos",
        "eef_quat": "robot0_eef_quat",
        "gripper_qpos": "robot0_gripper_qpos"
    }

    for k in obs_keys:
        # 1. Try the original key
        # 2. Try the mapped key
        # 3. Try adding 'robot0_' prefix
        actual_key = k
        if k not in obs_dict:
            actual_key = mapping.get(k, f"robot0_{k}")

        if actual_key in obs_dict:
            v = np.asarray(obs_dict[actual_key], dtype=np.float32)
            parts.append(v.ravel())
        else:
            raise KeyError(f"Observation key '{k}' (mapped to '{actual_key}') not found. "
                           f"Available keys: {list(obs_dict.keys())}")
    
    return np.concatenate(parts)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    saved_args = ckpt["args"]
    stats      = ckpt["stats"]

    obs_keys  = stats["obs_keys"]
    obs_dim   = stats["obs_dim"]
    act_dim   = stats["act_dim"]
    obs_mean  = torch.tensor(stats["obs_mean"], dtype=torch.float32)
    obs_std   = torch.tensor(stats["obs_std"],  dtype=torch.float32)
    act_mean  = np.array(stats["act_mean"], dtype=np.float32)
    act_std   = np.array(stats["act_std"],  dtype=np.float32)

    device = "cpu"   # evaluation is fast; no need for GPU

    policy = build_policy(
        obs_dim     = obs_dim,
        act_dim     = act_dim,
        policy_type = saved_args["policy"],
        hidden_dim  = saved_args["hidden_dim"],
        n_layers    = saved_args["n_layers"],
        dropout     = 0.0,        # no dropout at test time
    ).to(device)

    policy.load_state_dict(ckpt["model_state"])
    policy.eval()

    print(f"\n  Loaded checkpoint: {args.ckpt}  (val_loss={ckpt['val_loss']:.5f})")
    print(f"  obs_keys : {obs_keys}")
    print(f"  obs_dim  : {obs_dim}  |  act_dim: {act_dim}")

    # ------------------------------------------------------------------
    # Build environment
    # ------------------------------------------------------------------
    try:
        import robosuite as suite
    except ImportError:
        raise ImportError(
            "robosuite is not installed. "
            "Run:  pip install robosuite"
        )

    # Try to recover env_name / env_args from the HDF5 file used during training
    env_name = "PickPlace"   # default fallback
    env_args = {}

    hdf5_path = saved_args.get("hdf5", None)
    if hdf5_path and Path(hdf5_path).exists():
        import h5py
        with h5py.File(hdf5_path, "r") as f:
            if "env_name" in f["data"].attrs:
                env_name = f["data"].attrs["env_name"]
            if "env_args" in f["data"].attrs:
                raw = f["data"].attrs["env_args"]
                env_args = json.loads(raw) if isinstance(raw, str) else dict(raw)
                # env_args may nest robosuite_version / env_kwargs
                if "env_kwargs" in env_args:
                    env_args = env_args["env_kwargs"]

    np.random.seed(args.seed)
    env = build_env(env_name, env_args, obs_keys, render=args.render)
    print(f"\n  Environment: {env_name}")

    # ------------------------------------------------------------------
    # Video writer setup
    # ------------------------------------------------------------------
    video_writer = None
    if args.record_video:
        try:
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                args.record_video, fourcc, 20,
                (args.video_width, args.video_height),
            )
            print(f"  Recording video → {args.record_video}")
        except ImportError:
            print("  Warning: cv2 not installed; skipping video recording.")

    # ------------------------------------------------------------------
    # Rollout loop
    # ------------------------------------------------------------------
    returns, successes, lengths = [], [], []

    for ep in range(args.n_rollouts):
        obs_dict = env.reset()
        ep_return, ep_len, done = 0.0, 0, False

        while not done and ep_len < args.max_ep_len:
            # Build obs vector and normalise
            obs_vec = extract_obs_vector(obs_dict, obs_keys)
            obs_t   = (torch.from_numpy(obs_vec) - obs_mean) / obs_std
            obs_t   = obs_t.unsqueeze(0)              # (1, obs_dim)

            with torch.no_grad():
                if saved_args["policy"] == "stochastic":
                    action_t = policy.predict(obs_t, deterministic=True)
                else:
                    action_t = policy(obs_t)

            action_norm = action_t.squeeze(0).numpy()
            action      = action_norm * act_std + act_mean   # denormalise

            obs_dict, reward, done, info = env.step(action)

            if args.render:
                env.render()

            if video_writer is not None and ep == 0:
                frame = env.sim.render(
                    camera_name=args.camera,
                    width=args.video_width,
                    height=args.video_height,
                )[::-1]   # flip vertically
                import cv2
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            ep_return += reward
            ep_len    += 1

        success = bool(info.get("success", False))
        returns.append(ep_return)
        successes.append(success)
        lengths.append(ep_len)

        status = "✓" if success else "✗"
        print(f"  ep {ep+1:3d}/{args.n_rollouts}  {status}  "
              f"return={ep_return:.2f}  len={ep_len}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"  Success rate : {np.mean(successes)*100:.1f}%  "
          f"({int(np.sum(successes))}/{args.n_rollouts})")
    print(f"  Mean return  : {np.mean(returns):.3f} ± {np.std(returns):.3f}")
    print(f"  Mean ep len  : {np.mean(lengths):.1f}")
    print(f"{'='*50}\n")

    if video_writer is not None:
        video_writer.release()
        print(f"  Video saved to: {args.record_video}")

    env.close()


if __name__ == "__main__":
    main()