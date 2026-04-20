import torch
import numpy as np
import os
from pathlib import Path

# Import your teleop/env components
from envs.registry import EnvironmentRegistry
from model import build_policy

def main():
    # 1. Load Checkpoint
    ckpt_path = "best_policy.pt" # Update path if needed
    ckpt = torch.load(ckpt_path, map_location="cpu")
    saved_args = ckpt["args"]
    stats = ckpt["stats"]
    
    obs_mean = torch.tensor(stats["obs_mean"], dtype=torch.float32)
    obs_std = torch.tensor(stats["obs_std"], dtype=torch.float32)
    act_mean = np.array(stats["act_mean"], dtype=np.float32)
    act_std = np.array(stats["act_std"], dtype=np.float32)

    # 2. Build Policy
    policy = build_policy(
        obs_dim=stats["obs_dim"],
        act_dim=stats["act_dim"],
        policy_type=saved_args["policy"],
        hidden_dim=saved_args["hidden_dim"],
        n_layers=saved_args["n_layers"]
    )
    policy.load_state_dict(ckpt["model_state"])
    policy.eval()

    # 3. Setup Environment (Using your Registry logic)
    # Mapping 'PickPlace' to the specific object you need
    env_name = "PickPlace" 
    
    # Use the registry to get the environment class
    env_cls = EnvironmentRegistry.get_env_class(env_name)
    env = env_cls(
        robots="Panda",             # Match your training robot
        has_renderer=True,          # Show the window
        has_offscreen_renderer=True,
        use_camera_obs=False,
        render_gpu_device_id=0,
        control_freq=20,
    )

    print(f"Environment {env_name} started. Testing policy...")

    # 4. Evaluation Loop
    for ep in range(5):
        obs = env.reset()
        done = False
        ep_return = 0

        while not done:
            # Extract observation (using the mapping logic we discussed)
            # We must build the vector EXACTLY as the model saw it during training
            obs_list = []
            for k in stats["obs_keys"]:
                # Logic to handle 'robot0_' prefixes if the model expects them
                key = k if k in obs else f"robot0_{k}"
                if "egg" in key and key not in obs:
                    # Map to whatever object is actually in the scene (e.g., Bread)
                    key = "Bread_pos" if "pos" in key else "Bread_quat"
                
                obs_list.append(np.asarray(obs[key]).ravel())
            
            obs_vec = np.concatenate(obs_list)
            obs_t = (torch.from_numpy(obs_vec).float() - obs_mean) / obs_std
            
            with torch.no_grad():
                action_t = policy(obs_t.unsqueeze(0))
            
            # Denormalize and step
            action = action_t.squeeze(0).numpy() * act_std + act_mean
            obs, reward, done, info = env.step(action)
            
            env.render()
            ep_return += reward

        print(f"Episode {ep} finished. Return: {ep_return}")

if __name__ == "__main__":
    main()