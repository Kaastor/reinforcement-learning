"""
Common utilities for reinforcement learning agents.

This module implements shared functionality like ε-greedy action selection,
decay schedules, and logging utilities following Sutton & Barto conventions.
"""

import logging
import random
from typing import Callable, Optional

from mdp.core import State, Action


class EpsilonGreedyPolicy:
    """
    ε-greedy action selection for exploration-exploitation balance.
    
    With probability ε: select random action (exploration)
    With probability 1-ε: select greedy action (exploitation)
    """
    
    def __init__(
        self,
        epsilon: float = 0.1,
        epsilon_decay: Optional[Callable[[int], float]] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize ε-greedy policy.
        
        Args:
            epsilon: Initial exploration probability ε ∈ [0,1]
            epsilon_decay: Function that takes episode number and returns new ε
            random_seed: Random seed for reproducibility
        """
        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.episode_count = 0
        
        if random_seed is not None:
            random.seed(random_seed)
    
    def select_action(
        self, 
        state: State,
        available_actions: list[Action], 
        q_values: dict[tuple[State, Action], float]
    ) -> Action:
        """
        Select action using ε-greedy policy.
        
        Args:
            state: Current state
            available_actions: Available actions in this state
            q_values: Current Q-value estimates
            
        Returns:
            Selected action
        """
        if not available_actions:
            raise ValueError("No available actions")
        
        # Exploration: random action with probability ε
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        
        # Exploitation: greedy action (break ties randomly)
        best_value = float('-inf')
        best_actions = []
        
        for action in available_actions:
            q_value = q_values.get((state, action), 0.0)
            if q_value > best_value:
                best_value = q_value
                best_actions = [action]
            elif q_value == best_value:
                best_actions.append(action)
        
        return random.choice(best_actions)
    
    def update_epsilon(self, episode: int) -> None:
        """Update ε based on episode number."""
        self.episode_count = episode
        if self.epsilon_decay:
            self.epsilon = self.epsilon_decay(episode)
    
    def reset(self) -> None:
        """Reset ε to initial value."""
        self.epsilon = self.initial_epsilon
        self.episode_count = 0


class DecaySchedules:
    """Common decay schedules for hyperparameters."""
    
    @staticmethod
    def linear_decay(initial_value: float, final_value: float, decay_episodes: int) -> Callable[[int], float]:
        """
        Linear decay from initial_value to final_value over decay_episodes.
        
        Args:
            initial_value: Starting value
            final_value: Final value (reached at decay_episodes)
            decay_episodes: Number of episodes for complete decay
            
        Returns:
            Decay function that takes episode number and returns current value
        """
        def decay_fn(episode: int) -> float:
            if episode >= decay_episodes:
                return final_value
            
            progress = episode / decay_episodes
            return initial_value + (final_value - initial_value) * progress
        
        return decay_fn
    
    @staticmethod
    def exponential_decay(initial_value: float, decay_rate: float) -> Callable[[int], float]:
        """
        Exponential decay: value_t = initial_value * decay_rate^t.
        
        Args:
            initial_value: Starting value
            decay_rate: Decay rate ∈ (0,1)
            
        Returns:
            Decay function
        """
        def decay_fn(episode: int) -> float:
            return initial_value * (decay_rate ** episode)
        
        return decay_fn
    
    @staticmethod
    def step_decay(initial_value: float, decay_factor: float, step_size: int) -> Callable[[int], float]:
        """
        Step decay: reduce by decay_factor every step_size episodes.
        
        Args:
            initial_value: Starting value
            decay_factor: Multiplicative factor ∈ (0,1)
            step_size: Episodes between decay steps
            
        Returns:
            Decay function
        """
        def decay_fn(episode: int) -> float:
            num_decays = episode // step_size
            return initial_value * (decay_factor ** num_decays)
        
        return decay_fn


class LearningRateSchedule:
    """Adaptive learning rate schedules for better convergence."""
    
    @staticmethod
    def constant(alpha: float) -> Callable[[int], float]:
        """Constant learning rate."""
        return lambda episode: alpha
    
    @staticmethod
    def inverse_time(initial_alpha: float, decay_rate: float = 1.0) -> Callable[[int], float]:
        """
        Inverse time decay: α_t = initial_alpha / (1 + decay_rate * t).
        
        Ensures ∑ α_t = ∞ and ∑ α_t² < ∞ for convergence guarantees.
        """
        def schedule_fn(episode: int) -> float:
            return initial_alpha / (1.0 + decay_rate * episode)
        
        return schedule_fn
    
    @staticmethod
    def polynomial_decay(initial_alpha: float, power: float = 0.5) -> Callable[[int], float]:
        """
        Polynomial decay: α_t = initial_alpha / (t + 1)^power.
        
        Common choice is power = 0.5 for theoretical guarantees.
        """
        def schedule_fn(episode: int) -> float:
            return initial_alpha / ((episode + 1) ** power)
        
        return schedule_fn


def setup_agent_logging(
    name: str,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up standardized logging for RL agents.
    
    Args:
        name: Logger name
        level: Logging level
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(level)
    
    # Set format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger


def compute_mse(predicted: dict, target: dict) -> float:
    """
    Compute Mean Squared Error between two value functions.
    
    Args:
        predicted: Predicted values {state: value}
        target: Target values {state: value}
        
    Returns:
        MSE across all states
    """
    if not predicted or not target:
        return 0.0
    
    common_states = set(predicted.keys()) & set(target.keys())
    if not common_states:
        return 0.0
    
    squared_errors = []
    for state in common_states:
        error = predicted[state] - target[state]
        squared_errors.append(error ** 2)
    
    return sum(squared_errors) / len(squared_errors)


def compute_rmse(predicted: dict, target: dict) -> float:
    """Compute Root Mean Squared Error."""
    import math
    return math.sqrt(compute_mse(predicted, target))


def compute_max_error(predicted: dict, target: dict) -> float:
    """
    Compute maximum absolute error between value functions.
    
    Args:
        predicted: Predicted values
        target: Target values
        
    Returns:
        Maximum absolute error
    """
    if not predicted or not target:
        return 0.0
    
    common_states = set(predicted.keys()) & set(target.keys())
    if not common_states:
        return 0.0
    
    max_error = 0.0
    for state in common_states:
        error = abs(predicted[state] - target[state])
        max_error = max(max_error, error)
    
    return max_error