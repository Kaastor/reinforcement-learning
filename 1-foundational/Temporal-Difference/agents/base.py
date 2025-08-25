"""
Abstract base agent class for reinforcement learning algorithms.

This module provides the foundation for all RL agents following
Sutton & Barto conventions and notation.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from mdp.core import State, Action, Transition, MDP


class RLAgent(ABC):
    """
    Abstract base class for reinforcement learning agents.
    
    Defines the common interface for all RL algorithms including
    policy evaluation (TD(0), MC) and control (SARSA, Q-learning).
    """
    
    def __init__(
        self,
        mdp: MDP,
        alpha: float = 0.1,
        gamma: float = 1.0,
        name: str = "RLAgent"
    ):
        """
        Initialize base agent.
        
        Args:
            mdp: The MDP environment
            alpha: Learning rate α ∈ (0,1]
            gamma: Discount factor γ ∈ [0,1]
            name: Agent name for logging
        """
        self.mdp = mdp
        self.alpha = alpha
        self.gamma = gamma
        self.name = name
        
        # Set up logging
        self.logger = logging.getLogger(f"agents.{name}")
        
        # Initialize tracking
        self.episode_count = 0
        self.step_count = 0
        self.training_history = []
        
        # Initialize value functions (to be implemented by subclasses)
        self.reset()
    
    @abstractmethod
    def reset(self) -> None:
        """Reset agent's internal state (value functions, etc.)."""
        pass
    
    @abstractmethod
    def update(self, transition: Transition) -> float:
        """
        Update agent with a single transition.
        
        Args:
            transition: (S_t, A_t, R_{t+1}, S_{t+1}, done)
            
        Returns:
            Update magnitude (e.g., |δ_t| for TD methods)
        """
        pass
    
    @abstractmethod
    def get_value(self, state: State) -> float:
        """Get value estimate for a state."""
        pass
    
    def train_episode(self, episode: list[Transition]) -> Dict[str, float]:
        """
        Train agent on a complete episode.
        
        Args:
            episode: List of transitions
            
        Returns:
            Training metrics for this episode
        """
        total_reward = 0.0
        total_update = 0.0
        
        for transition in episode:
            update_magnitude = self.update(transition)
            total_reward += transition.reward
            total_update += abs(update_magnitude)
            self.step_count += 1
        
        self.episode_count += 1
        
        # Log episode metrics
        metrics = {
            "episode": self.episode_count,
            "steps": len(episode),
            "total_reward": total_reward,
            "avg_update": total_update / len(episode) if episode else 0.0
        }
        
        self.training_history.append(metrics)
        self.logger.debug(f"Episode {self.episode_count}: {metrics}")
        
        return metrics
    
    def get_training_history(self) -> list[Dict[str, float]]:
        """Get complete training history."""
        return self.training_history.copy()
    
    def save_checkpoint(self) -> Dict[str, Any]:
        """Save agent state for checkpointing."""
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "episode_count": self.episode_count,
            "step_count": self.step_count,
            "training_history": self.training_history
        }
    
    def load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Load agent state from checkpoint."""
        self.alpha = checkpoint["alpha"]
        self.gamma = checkpoint["gamma"]
        self.episode_count = checkpoint["episode_count"]
        self.step_count = checkpoint["step_count"]
        self.training_history = checkpoint["training_history"]


class ValueBasedAgent(RLAgent):
    """
    Base class for value-based agents (TD(0), Monte Carlo).
    
    These agents learn value functions V(s) for policy evaluation.
    """
    
    def __init__(self, mdp: MDP, alpha: float = 0.1, gamma: float = 1.0, name: str = "ValueAgent"):
        self.V: Dict[State, float] = {}
        super().__init__(mdp, alpha, gamma, name)
    
    def reset(self) -> None:
        """Reset value function."""
        self.V = {state: 0.0 for state in self.mdp.get_states()}
    
    def get_value(self, state: State) -> float:
        """Get value estimate V(s)."""
        return self.V.get(state, 0.0)
    
    def set_value(self, state: State, value: float) -> None:
        """Set value estimate V(s)."""
        self.V[state] = value
    
    def get_all_values(self) -> Dict[State, float]:
        """Get all value estimates."""
        return self.V.copy()


class ActionValueBasedAgent(RLAgent):
    """
    Base class for action-value based agents (SARSA, Q-learning).
    
    These agents learn action-value functions Q(s,a) for control.
    """
    
    def __init__(self, mdp: MDP, alpha: float = 0.1, gamma: float = 1.0, name: str = "QAgent"):
        super().__init__(mdp, alpha, gamma, name)
        self.Q: Dict[tuple[State, Action], float] = {}
    
    def reset(self) -> None:
        """Reset action-value function."""
        self.Q = {}
        for state in self.mdp.get_states():
            if not self.mdp.is_terminal(state):
                for action in self.mdp.get_actions(state):
                    self.Q[(state, action)] = 0.0
    
    def get_value(self, state: State) -> float:
        """Get state value V(s) = max_a Q(s,a)."""
        if self.mdp.is_terminal(state):
            return 0.0
        
        actions = self.mdp.get_actions(state)
        if not actions:
            return 0.0
        
        q_values = [self.get_q_value(state, action) for action in actions]
        return max(q_values) if q_values else 0.0
    
    def get_q_value(self, state: State, action: Action) -> float:
        """Get action-value estimate Q(s,a)."""
        return self.Q.get((state, action), 0.0)
    
    def set_q_value(self, state: State, action: Action, value: float) -> None:
        """Set action-value estimate Q(s,a)."""
        self.Q[(state, action)] = value
    
    def get_all_q_values(self) -> Dict[tuple[State, Action], float]:
        """Get all Q-value estimates."""
        return self.Q.copy()
    
    @abstractmethod
    def select_action(self, state: State, available_actions: list[Action]) -> Action:
        """Select action using agent's policy (e.g., ε-greedy)."""
        pass