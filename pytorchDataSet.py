import h5py, torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class RobotDemoDataset(Dataset):
    def __init__(self, hdf5_path, obs_keys, chunk_size = 1):
        self.data = []
        with h5py.File(hdf5_path, "r") as f:
            for demo_key in f["data"].keys():
                demo = f["data"][demo_key]
                T = demo["actions"].shape[0]

                obs_list = [demo["obs"][k][:] for k in obs_keys]
                obs = np.concatenate(obs_list, axis = 1)
                actions = demo["actions"][:]

                for t in range(T - chunk_size):
                    self.data.append({
                        "obs": obs[t],
                        "action": actions[t],
                    })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return (
            torch.tensor(sample["obs"], dtype = torch.float32),
            torch.tensor(sample["action"], dtype = torch.float32)
        )

obs_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
dataset = RobotDemoDataset("pick_egg_safe_combined.hdf5", obs_keys)
loader = DataLoader(dataset, batch_size=256, shuffle = True, num_workers = 4)