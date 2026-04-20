"""
dataset.py — HDF5 demonstration dataset for Behavioral Cloning.

Robosuite HDF5 layout expected:
  data/
    demo_0/
      obs/
        <key_1>  (T, ...)   e.g. robot0_eef_pos, robot0_eef_quat,
        <key_2>             robot0_gripper_qpos, object-state, ...
      actions   (T, A)
    demo_1/
      ...
  attrs:
    env_name    str
    env_args    json str   (robosuite env kwargs)
    total       int        (number of demos)
"""

import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_demo_keys(f: h5py.File) -> list[str]:
    """Return sorted list of demo group names under 'data/'."""
    return sorted(f["data"].keys())


def flatten_obs(obs_group: h5py.Group, obs_keys: list[str]) -> np.ndarray:
    """Concatenate selected observation arrays along the feature axis."""
    parts = []
    for k in obs_keys:
        arr = obs_group[k][:]           # (T, ...)
        if arr.ndim == 1:
            arr = arr[:, None]          # (T, 1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)   # flatten spatial dims
        parts.append(arr.astype(np.float32))
    return np.concatenate(parts, axis=-1)   # (T, obs_dim)


# ---------------------------------------------------------------------------
# Main Dataset
# ---------------------------------------------------------------------------

class RobosuiteHDF5Dataset(Dataset):
    """
    Flat dataset: each item is a (obs, action) pair drawn from all demos.

    Parameters
    ----------
    hdf5_path   : path to the .hdf5 file
    obs_keys    : list of observation keys to include.
                  Pass None to auto-detect all scalar/vector keys
                  (excludes image arrays with >2 dims if skip_images=True).
    skip_images : when obs_keys is None, skip high-dim image observations
    normalize   : if True, compute per-feature mean/std and normalise obs + actions
    demo_limit  : only load the first N demos (useful for quick debugging)
    """

    def __init__(
        self,
        hdf5_path: str,
        obs_keys: list[str] | None = None,
        skip_images: bool = True,
        normalize: bool = True,
        demo_limit: int | None = None,
    ):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.normalize = normalize

        all_obs: list[np.ndarray] = []
        all_actions: list[np.ndarray] = []

        with h5py.File(hdf5_path, "r") as f:
            demo_keys = get_demo_keys(f)
            if demo_limit is not None:
                demo_keys = demo_keys[:demo_limit]

            # Auto-detect obs keys from first demo if not provided
            if obs_keys is None:
                obs_group_0 = f["data"][demo_keys[0]]["obs"]
                obs_keys = []
                for k in sorted(obs_group_0.keys()):
                    ds = obs_group_0[k]
                    if skip_images and ds.ndim > 2:
                        print(f"  [dataset] Skipping image key: {k} {ds.shape}")
                        continue
                    obs_keys.append(k)
                print(f"  [dataset] Auto-detected obs keys: {obs_keys}")

            self.obs_keys = obs_keys

            for dk in demo_keys:
                demo = f["data"][dk]
                obs = flatten_obs(demo["obs"], obs_keys)   # (T, obs_dim)
                actions = demo["actions"][:].astype(np.float32)  # (T, act_dim)

                # Some demos store one extra obs at the end; align lengths
                T = min(len(obs), len(actions))
                all_obs.append(obs[:T])
                all_actions.append(actions[:T])

        obs_arr = np.concatenate(all_obs, axis=0)       # (N, obs_dim)
        act_arr = np.concatenate(all_actions, axis=0)   # (N, act_dim)

        self.obs_dim = obs_arr.shape[-1]
        self.act_dim = act_arr.shape[-1]

        # Normalisation statistics
        if normalize:
            self.obs_mean  = obs_arr.mean(0)
            self.obs_std   = obs_arr.std(0).clip(min=1e-6)
            self.act_mean  = act_arr.mean(0)
            self.act_std   = act_arr.std(0).clip(min=1e-6)
            obs_arr  = (obs_arr  - self.obs_mean)  / self.obs_std
            act_arr  = (act_arr  - self.act_mean)  / self.act_std
        else:
            self.obs_mean = np.zeros(self.obs_dim, dtype=np.float32)
            self.obs_std  = np.ones(self.obs_dim,  dtype=np.float32)
            self.act_mean = np.zeros(self.act_dim, dtype=np.float32)
            self.act_std  = np.ones(self.act_dim,  dtype=np.float32)

        self.obs     = torch.from_numpy(obs_arr)
        self.actions = torch.from_numpy(act_arr)

        print(
            f"  [dataset] Loaded {len(demo_keys)} demos | "
            f"{len(self.obs)} transitions | "
            f"obs_dim={self.obs_dim} | act_dim={self.act_dim}"
        )

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]

    # ------------------------------------------------------------------
    # Helpers for denormalising actions at inference time
    # ------------------------------------------------------------------
    def denorm_action(self, action: np.ndarray) -> np.ndarray:
        return action * self.act_std + self.act_mean

    def norm_obs(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.obs_mean) / self.obs_std

    def get_stats(self) -> dict:
        return {
            "obs_mean": self.obs_mean,
            "obs_std":  self.obs_std,
            "act_mean": self.act_mean,
            "act_std":  self.act_std,
            "obs_keys": self.obs_keys,
            "obs_dim":  self.obs_dim,
            "act_dim":  self.act_dim,
        }


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_dataloaders(
    hdf5_path: str,
    val_fraction: float = 0.1,
    batch_size: int = 256,
    num_workers: int = 4,
    obs_keys: list[str] | None = None,
    normalize: bool = True,
    demo_limit: int | None = None,
    seed: int = 42,
):
    """Return (train_loader, val_loader, dataset_stats)."""
    dataset = RobosuiteHDF5Dataset(
        hdf5_path,
        obs_keys=obs_keys,
        normalize=normalize,
        demo_limit=demo_limit,
    )

    n_val   = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, dataset.get_stats()