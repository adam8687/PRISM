"""
model.py — Behavioural Cloning policy network.

Architecture: a residual MLP with LayerNorm and GELU activations.
Optionally outputs a Gaussian distribution (stochastic BC) rather than
a deterministic point estimate.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Pre-norm residual block: LayerNorm → Linear → GELU → Dropout → Linear + skip."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)


# ---------------------------------------------------------------------------
# Deterministic BC (MSE loss)
# ---------------------------------------------------------------------------

class BCPolicyMLP(nn.Module):
    """
    Deterministic MLP policy: obs → action.

    Parameters
    ----------
    obs_dim     : dimensionality of (normalised) observation vector
    act_dim     : dimensionality of action vector
    hidden_dim  : width of hidden layers
    n_layers    : number of residual blocks (depth)
    dropout     : dropout rate inside residual blocks
    act_limit   : if provided, tanh-squash output to [-act_limit, act_limit]
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        dropout: float = 0.0,
        act_limit: float | None = None,
    ):
        super().__init__()
        self.act_limit = act_limit

        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_layers)]
        )
        self.output_head = nn.Linear(hidden_dim, act_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, obs_dim)  →  action: (B, act_dim)"""
        h = self.input_proj(obs)
        h = self.res_blocks(h)
        a = self.output_head(h)
        if self.act_limit is not None:
            a = torch.tanh(a) * self.act_limit
        return a

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, obs: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self(obs)


# ---------------------------------------------------------------------------
# Stochastic BC (NLL loss, Gaussian policy)
# ---------------------------------------------------------------------------

class StochasticBCPolicy(nn.Module):
    """
    Gaussian policy: obs → (mu, log_std) → Normal(mu, exp(log_std)).

    Training loss: negative log-likelihood.
    Inference: return mean (or sample).
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX =  2.0

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_layers)],
        )
        self.mu_head      = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor):
        """Returns a Normal distribution."""
        h       = self.trunk(obs)
        mu      = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return Normal(mu, log_std.exp())

    def nll_loss(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        dist = self(obs)
        return -dist.log_prob(actions).sum(-1).mean()

    @torch.no_grad()
    def predict(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        self.eval()
        dist = self(obs)
        return dist.mean if deterministic else dist.sample()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_policy(
    obs_dim: int,
    act_dim: int,
    policy_type: str = "deterministic",   # "deterministic" | "stochastic"
    hidden_dim: int = 256,
    n_layers: int = 3,
    dropout: float = 0.0,
    act_limit: float | None = None,
) -> nn.Module:
    if policy_type == "deterministic":
        return BCPolicyMLP(obs_dim, act_dim, hidden_dim, n_layers, dropout, act_limit)
    elif policy_type == "stochastic":
        return StochasticBCPolicy(obs_dim, act_dim, hidden_dim, n_layers, dropout)
    else:
        raise ValueError(f"Unknown policy_type: {policy_type!r}")