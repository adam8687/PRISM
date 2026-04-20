"""
train.py — Train a Behavioural Cloning policy from robosuite HDF5 demonstrations.

Usage
-----
  python train.py                              # all defaults
  python train.py --hdf5 my_demos.hdf5        # custom data path
  python train.py --policy stochastic         # Gaussian policy
  python train.py --epochs 200 --lr 3e-4

The script saves:
  checkpoints/best_policy.pt   ← best val-loss checkpoint
  checkpoints/last_policy.pt   ← final checkpoint
  checkpoints/stats.npz        ← normalisation statistics needed at eval time
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import make_dataloaders
from model import build_policy


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="BC training for robosuite demos")
    p.add_argument("--hdf5",        default="pick_egg_safe_hdf5_50.hdf5")
    p.add_argument("--out_dir",     default="checkpoints")
    p.add_argument("--policy",      default="deterministic",
                   choices=["deterministic", "stochastic"])
    p.add_argument("--hidden_dim",  type=int,   default=256)
    p.add_argument("--n_layers",    type=int,   default=3)
    p.add_argument("--dropout",     type=float, default=0.05)
    p.add_argument("--batch_size",  type=int,   default=256)
    p.add_argument("--epochs",      type=int,   default=300)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--val_frac",    type=float, default=0.1)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--log_every",   type=int,   default=10,
                   help="Print a status line every N epochs")
    # Optionally restrict which observation keys to use
    p.add_argument("--obs_keys",    nargs="*",  default=None,
                   help="Observation keys to include (default: auto-detect all non-image)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_loss(
    policy: nn.Module,
    obs: torch.Tensor,
    actions: torch.Tensor,
    policy_type: str,
) -> torch.Tensor:
    if policy_type == "stochastic":
        return policy.nll_loss(obs, actions)
    else:
        pred = policy(obs)
        return nn.functional.mse_loss(pred, actions)


@torch.no_grad()
def evaluate(policy, loader, policy_type, device):
    policy.eval()
    total_loss, total_n = 0.0, 0
    for obs, actions in loader:
        obs, actions = obs.to(device), actions.to(device)
        loss = compute_loss(policy, obs, actions, policy_type)
        total_loss += loss.item() * len(obs)
        total_n    += len(obs)
    return total_loss / total_n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps"  if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\n{'='*60}")
    print(f"  Behavioural Cloning — robosuite egg pick")
    print(f"  device  : {device}")
    print(f"  policy  : {args.policy}")
    print(f"  data    : {args.hdf5}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("[1/3] Loading data …")
    train_loader, val_loader, stats = make_dataloaders(
        hdf5_path    = args.hdf5,
        val_fraction = args.val_frac,
        batch_size   = args.batch_size,
        num_workers  = args.num_workers,
        obs_keys     = args.obs_keys,
        normalize    = True,
        seed         = args.seed,
    )
    obs_dim = stats["obs_dim"]
    act_dim = stats["act_dim"]

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print("[2/3] Building policy …")
    policy = build_policy(
        obs_dim    = obs_dim,
        act_dim    = act_dim,
        policy_type= args.policy,
        hidden_dim = args.hidden_dim,
        n_layers   = args.n_layers,
        dropout    = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Optimiser + scheduler
    # ------------------------------------------------------------------
    optimiser = AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print("[3/3] Training …\n")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(1, args.epochs + 1):
        policy.train()
        epoch_loss, epoch_n = 0.0, 0
        t0 = time.time()

        for obs, actions in train_loader:
            obs, actions = obs.to(device), actions.to(device)

            optimiser.zero_grad()
            loss = compute_loss(policy, obs, actions, args.policy)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimiser.step()

            epoch_loss += loss.item() * len(obs)
            epoch_n    += len(obs)

        scheduler.step()

        train_loss = epoch_loss / epoch_n
        val_loss   = evaluate(policy, val_loader, args.policy, device)
        lr_now     = scheduler.get_last_lr()[0]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(lr_now)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch":      epoch,
                    "model_state": policy.state_dict(),
                    "val_loss":   val_loss,
                    "args":       vars(args),
                    "stats":      stats,
                },
                out_dir / "best_policy.pt",
            )

        if epoch % args.log_every == 0 or epoch == 1:
            dt = time.time() - t0
            marker = " ← best" if val_loss == best_val_loss else ""
            print(
                f"  epoch {epoch:4d}/{args.epochs}"
                f"  train={train_loss:.5f}"
                f"  val={val_loss:.5f}"
                f"  lr={lr_now:.2e}"
                f"  ({dt:.1f}s){marker}"
            )

    # Save last checkpoint
    torch.save(
        {
            "epoch":      args.epochs,
            "model_state": policy.state_dict(),
            "val_loss":   val_loss,
            "args":       vars(args),
            "stats":      stats,
        },
        out_dir / "last_policy.pt",
    )

    # Save normalisation stats separately (convenient for eval)
    np.savez(
        out_dir / "stats.npz",
        obs_mean  = stats["obs_mean"],
        obs_std   = stats["obs_std"],
        act_mean  = stats["act_mean"],
        act_std   = stats["act_std"],
        obs_keys  = np.array(stats["obs_keys"]),
    )

    # Save loss history
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Done. Best val loss: {best_val_loss:.5f}")
    print(f"  Checkpoints saved to: {out_dir}/")


if __name__ == "__main__":
    main()