"""Math module for reinforcement learning calculations."""

from .bellman import BellmanSolver
from .returns import (
    compute_monte_carlo_returns,
    compute_discounted_rewards, 
    compute_n_step_returns,
    compute_td_error,
    compute_gae_returns,
    compute_episode_return
)

__all__ = [
    "BellmanSolver",
    "compute_monte_carlo_returns",
    "compute_discounted_rewards",
    "compute_n_step_returns", 
    "compute_td_error",
    "compute_gae_returns",
    "compute_episode_return"
]