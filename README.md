# PRISM

PRISM is a small robotics imitation-learning project built around Robosuite.
The workflow is:

1. Collect demonstrations in simulation.
2. Save trajectories in HDF5 format.
3. Train a Behavioral Cloning (BC) policy with PyTorch.
4. Evaluate the policy in simulation.


## Tech Stack

- Python 3.10
- Robosuite + MuJoCo (robot simulation)
- PyTorch (policy training)
- HDF5 / h5py (dataset storage)
- NumPy (numerical ops)


## Repository Layout

- `safe_demos_collection.py`: keyboard teleoperation demo collection (Lift task)
- `teleop.py`: manual keyboard control loop for Robosuite
- `main.py`: scripted Lift baseline
- `BC_Policy/`: BC dataset/model/train/eval code
- `final_BC_policy.py`: all-in-one BC training + rollout script for PickEgg-style data


## Setup

Create and activate a Python 3.10 environment, then install dependencies.

```bash
pip install mujoco robosuite torch numpy h5py opencv-python tqdm
```

If your environment uses a custom benchmark wrapper (`oopsiebench`), install that package in the same environment.


## Collect Demonstrations

Run keyboard data collection:

```bash
python safe_demos_collection.py
```

Default output is `safe_demos.hdf5`.


## Train a BC Policy

From the policy folder:

```bash
cd BC_Policy
python train.py --hdf5 ../safe_demos.hdf5 --out_dir checkpoints
```

Useful options:

```bash
python train.py --help
```


## Evaluate a Trained Policy

From `BC_Policy`:

```bash
python evaluate_policy.py --policy checkpoints/best_policy.pt --num_episodes 100
```


## Quick Alternative Pipeline

`final_BC_policy.py` includes a full train + evaluate flow in one script.
Before running it, update `DATASET_PATH` to your local dataset path.

```bash
python final_BC_policy.py
```


## Notes

- Dataset schema matters: training code expects Robosuite-style HDF5 groups under `data/demo_x/...`.
- Several scripts target different tasks/environments (Lift vs PickEgg). Keep dataset and evaluation environment aligned.


