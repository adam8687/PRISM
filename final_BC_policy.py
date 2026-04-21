import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from robosuite.controllers import load_composite_controller_config

from oopsiebench.envs.registry import EnvironmentRegistry


DATASET_PATH = Path("/home/prism/PRISM/PRISM/PRISM/pick_egg_safe_hdf5_50.hdf5")
ROLLOUT_VIDEO_DIR = Path("pick_egg_bc_rollouts")
ENV_NAME = "pick_egg"
SEED = 0
HIDDEN_DIM = 512
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 50
NUM_EVAL_EPISODES = 5
MAX_STEPS = 450
VIDEO_FPS = 20
CAMERA_WIDTH = 256
CAMERA_HEIGHT = 256
OBS_FEATURE_KEYS = ("robot0_proprio", "egg_pos", "egg_quat")
GRIPPER_ACTION_INDEX = 6
GRIPPER_OPEN_VALUE = -1.0
GRIPPER_CLOSE_VALUE = 0.15
# DEVICE = torch.device("mps")
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


class BCDataset(Dataset):
    def __init__(self, states: np.ndarray, actions: np.ndarray, gripper_targets: np.ndarray):
        self.states = torch.from_numpy(states.astype(np.float32))
        self.actions = torch.from_numpy(actions.astype(np.float32))
        self.gripper_targets = torch.from_numpy(gripper_targets.astype(np.float32))

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def __getitem__(self, index: int):
        return self.states[index], self.actions[index], self.gripper_targets[index]


class BCPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, action_dim),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states)


def load_bc_data(dataset_path: Path):
    all_features = []
    all_actions = []
    eval_demos = []

    with h5py.File(dataset_path, "r") as hdf:
        demo_names = sorted(hdf["data"].keys())
        for demo_name in demo_names:
            demo = hdf["data"][demo_name]
            full_states = demo["states"][:].astype(np.float64)
            full_actions = demo["actions"][:].astype(np.float64)
            obs_features = [demo["obs"][key][:].astype(np.float64) for key in OBS_FEATURE_KEYS]
            full_features = np.concatenate(obs_features, axis=1)

            # In this dataset, actions[0] has no preceding recorded state.
            # Align training pairs as (obs_t, action_{t+1}).
            features = full_features[:-1]
            actions = full_actions[1:]
            all_features.append(features)
            all_actions.append(actions)

            model_xml = demo.attrs["model_file"]
            ep_meta = json.loads(demo.attrs["ep_meta"])
            eval_demos.append((demo_name, model_xml, ep_meta, full_states[0]))

    stacked_features = np.concatenate(all_features, axis=0)
    stacked_actions = np.concatenate(all_actions, axis=0)
    return stacked_features, stacked_actions, eval_demos


def build_loader(features: np.ndarray, actions: np.ndarray, gripper_targets: np.ndarray):
    dataset = BCDataset(features, actions, gripper_targets)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    return loader


def compute_normalization(features: np.ndarray, actions: np.ndarray):
    feature_mean = features.mean(axis=0).astype(np.float32)
    feature_std = features.std(axis=0).astype(np.float32)
    feature_std[feature_std < 1e-6] = 1.0

    action_mean = actions.mean(axis=0).astype(np.float32)
    action_std = actions.std(axis=0).astype(np.float32)
    action_std[action_std < 1e-6] = 1.0

    return feature_mean, feature_std, action_mean, action_std


def normalize_numpy(values: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (values - mean) / std


def train_bc(
    features: np.ndarray,
    actions: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
):
    norm_features = normalize_numpy(features, feature_mean, feature_std)
    norm_actions = normalize_numpy(actions, action_mean, action_std)
    gripper_targets = (actions[:, GRIPPER_ACTION_INDEX] > 0.0).astype(np.float32)
    loader = build_loader(norm_features, norm_actions, gripper_targets)

    reg_indices = [i for i in range(actions.shape[1]) if i != GRIPPER_ACTION_INDEX]
    reg_indices_t = torch.tensor(reg_indices, dtype=torch.long, device=DEVICE)
    positive = float(np.sum(gripper_targets))
    negative = float(gripper_targets.shape[0] - positive)
    if positive > 0.0:
        pos_weight_value = negative / positive
    else:
        pos_weight_value = 1.0

    policy = BCPolicy(features.shape[1], actions.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    mse_loss_fn = nn.MSELoss()
    bce_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=DEVICE)
    )

    policy.train()
    for epoch_idx in range(NUM_EPOCHS):
        running_loss = 0.0
        num_batches = 0
        for batch_states, batch_actions, batch_gripper_targets in loader:
            batch_states = batch_states.to(DEVICE)
            batch_actions = batch_actions.to(DEVICE)
            batch_gripper_targets = batch_gripper_targets.to(DEVICE)
            pred = policy(batch_states)
            reg_loss = mse_loss_fn(pred[:, reg_indices_t], batch_actions[:, reg_indices_t])
            grip_loss = bce_loss_fn(pred[:, GRIPPER_ACTION_INDEX], batch_gripper_targets)
            loss = reg_loss + grip_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            num_batches += 1

        avg_loss = running_loss / max(num_batches, 1)
        print(f"Epoch {epoch_idx + 1}/{NUM_EPOCHS} - loss: {avg_loss:.6f}")

    return policy


def create_env():
    env_config = EnvironmentRegistry.get(ENV_NAME)
    env = env_config.damageable_class(
        robots=env_config.robot,
        controller_configs=load_composite_controller_config(robot=env_config.robot),
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=env_config.camera_name,
        camera_widths=CAMERA_WIDTH,
        camera_heights=CAMERA_HEIGHT,
        camera_depths=False,
        render_segmentation=False,
        control_freq=env_config.control_freq,
    )
    return env


def save_video(frames: list[np.ndarray], output_path: Path):
    if len(frames) == 0:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        VIDEO_FPS,
        (CAMERA_WIDTH, CAMERA_HEIGHT),
    )
    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()


def evaluate_and_record(
    env,
    eval_demos,
    policy: BCPolicy,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
):
    feature_mean_t = torch.from_numpy(feature_mean).to(DEVICE)
    feature_std_t = torch.from_numpy(feature_std).to(DEVICE)
    action_mean_t = torch.from_numpy(action_mean).to(DEVICE)
    action_std_t = torch.from_numpy(action_std).to(DEVICE)

    policy.eval()
    episode_returns = []
    episode_success = []
    image_key = f"{EnvironmentRegistry.get(ENV_NAME).camera_name}_image"

    for episode_idx in range(NUM_EVAL_EPISODES):
        demo_name, model_xml, ep_meta, initial_state = eval_demos[episode_idx]

        env.set_ep_meta(ep_meta)
        env.reset()
        env.reset_from_xml_string(model_xml)
        env.sim.reset()
        env.sim.set_state_from_flattened(initial_state)
        env.sim.forward()

        obs, _ = env.get_observations()
        frames = [obs[image_key][:, :, :3].astype(np.uint8)]
        total_reward = 0.0
        success = False

        for _ in range(MAX_STEPS):
            feature = np.concatenate(
                [obs[key].reshape(-1).astype(np.float32) for key in OBS_FEATURE_KEYS],
                axis=0,
            )
            feature_t = torch.from_numpy(feature).to(DEVICE)
            norm_feature_t = (feature_t - feature_mean_t) / feature_std_t
            with torch.no_grad():
                pred_norm_action_t = policy(norm_feature_t)
            action_t = pred_norm_action_t * action_std_t + action_mean_t
            action = action_t.detach().cpu().numpy().astype(np.float64)
            if float(pred_norm_action_t[GRIPPER_ACTION_INDEX].item()) > 0.0:
                action[GRIPPER_ACTION_INDEX] = GRIPPER_CLOSE_VALUE
            else:
                action[GRIPPER_ACTION_INDEX] = GRIPPER_OPEN_VALUE
            obs, reward, done, info = env.step(action)
            frames.append(obs[image_key][:, :, :3].astype(np.uint8))

            total_reward += float(reward)
            step_success = bool(done) or bool(env._check_success())
            if "success" in info:
                step_success = step_success or bool(info["success"])
            if "task_success" in info:
                step_success = step_success or bool(info["task_success"])
            if step_success:
                success = True
            if done:
                break

        episode_returns.append(total_reward)
        episode_success.append(success)
        video_path = ROLLOUT_VIDEO_DIR / f"episode_{episode_idx}.mp4"
        save_video(frames, video_path)
        print(
            f"Episode {episode_idx + 1}/{NUM_EVAL_EPISODES} - "
            f"success: {success} - return: {total_reward:.3f} - video: {video_path}"
        )

    return episode_returns, episode_success


def main():
    # if not torch.backends.mps.is_available():
    #     raise RuntimeError("MPS device is not available. This script requires MPS.")
    print(f"Using device: {DEVICE}")


    np.random.seed(SEED)
    torch.manual_seed(SEED)

    features, actions, eval_demos = load_bc_data(DATASET_PATH)
    feature_mean, feature_std, action_mean, action_std = compute_normalization(features, actions)
    policy = train_bc(features, actions, feature_mean, feature_std, action_mean, action_std)

    env = create_env()
    try:
        returns, successes = evaluate_and_record(
            env,
            eval_demos,
            policy,
            feature_mean,
            feature_std,
            action_mean,
            action_std,
        )
    finally:
        env.close()

    success_rate = float(np.mean(np.asarray(successes, dtype=np.float64)))
    print(f"Trained BC on {features.shape[0]} samples.")
    print(f"Saved rollout videos to: {ROLLOUT_VIDEO_DIR}")
    print(f"Device: {DEVICE}")
    print(f"Returns: {[round(r, 3) for r in returns]}")
    print(f"Successes: {successes} (success_rate={success_rate:.3f})")


if __name__ == "__main__":
    main()