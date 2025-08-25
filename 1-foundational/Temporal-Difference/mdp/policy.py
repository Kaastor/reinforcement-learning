"""
Policy representation and sampling following Sutton & Barto notation.

This module implements policy π(a|s) representations and episode generation
for MDP environments.
"""

import random
from abc import ABC, abstractmethod
from typing import Any

from .core import State, Action, Transition, MDP


class Policy(ABC):
    """
    Abstract base class for policies π(a|s).
    
    A policy defines the probability distribution over actions
    given a state: π(a|s) = P(A_t = a | S_t = s).
    """
    
    @abstractmethod
    def sample_action(self, state: State, available_actions: list[Action]) -> Action:
        """Sample an action from π(a|s)."""
        pass
    
    @abstractmethod
    def get_action_probability(self, state: State, action: Action, available_actions: list[Action]) -> float:
        """Get π(a|s) for a specific state-action pair."""
        pass


class UniformPolicy(Policy):
    """
    Uniform random policy: π(a|s) = 1/|A(s)| for all a ∈ A(s).
    
    Samples actions uniformly from the available action space.
    Useful for exploration and as a baseline policy.
    """
    
    def __init__(self, random_seed: int | None = None):
        """Initialize uniform policy with optional random seed."""
        if random_seed is not None:
            random.seed(random_seed)
    
    def sample_action(self, state: State, available_actions: list[Action]) -> Action:
        """Sample action uniformly: π(a|s) = 1/|A(s)|."""
        return random.choice(available_actions)
    
    def get_action_probability(self, state: State, action: Action, available_actions: list[Action]) -> float:
        """Return uniform probability 1/|A(s)|."""
        if action in available_actions:
            return 1.0 / len(available_actions)
        return 0.0


class DeterministicPolicy(Policy):
    """
    Deterministic policy: π(a|s) = 1 for specific a, 0 otherwise.
    
    Maps each state to a single action with probability 1.
    Useful for representing optimal policies and greedy policies.
    """
    
    def __init__(self, policy_map: dict[State, Action]):
        """
        Initialize with state->action mapping.
        
        Args:
            policy_map: Dictionary mapping states to actions
        """
        self.policy_map = policy_map
    
    def sample_action(self, state: State, available_actions: list[Action]) -> Action:
        """Return the deterministic action for this state."""
        if state not in self.policy_map:
            raise ValueError(f"No action defined for state {state}")
        
        action = self.policy_map[state]
        if action not in available_actions:
            raise ValueError(f"Action {action} not available in state {state}")
        
        return action
    
    def get_action_probability(self, state: State, action: Action, available_actions: list[Action]) -> float:
        """Return 1.0 for policy action, 0.0 otherwise."""
        if state not in self.policy_map:
            return 0.0
        
        return 1.0 if action == self.policy_map[state] else 0.0


class EpisodeGenerator:
    """
    Generates episodes by running a policy on an MDP environment.
    
    Creates sequences of transitions (S_t, A_t, R_{t+1}, S_{t+1})
    by following policy π in environment until termination.
    """
    
    def __init__(self, mdp: MDP, policy: Policy, max_steps: int = 1000):
        """
        Initialize episode generator.
        
        Args:
            mdp: The MDP environment
            policy: Policy to follow
            max_steps: Maximum steps per episode (prevents infinite loops)
        """
        self.mdp = mdp
        self.policy = policy
        self.max_steps = max_steps
    
    def generate_episode(self) -> list[Transition]:
        """
        Generate a single episode following the policy.
        
        Returns:
            List of transitions [(S_0, A_0, R_1, S_1), (S_1, A_1, R_2, S_2), ...]
        """
        episode = []
        state = self.mdp.reset()
        
        for step in range(self.max_steps):
            if self.mdp.is_terminal(state):
                break
            
            available_actions = self.mdp.get_actions(state)
            if not available_actions:
                break
            
            action = self.policy.sample_action(state, available_actions)
            transition = self.mdp.step(state, action)
            episode.append(transition)
            
            state = transition.next_state
            if transition.done or state is None:
                break
        
        return episode
    
    def generate_episodes(self, num_episodes: int) -> list[list[Transition]]:
        """
        Generate multiple episodes.
        
        Args:
            num_episodes: Number of episodes to generate
            
        Returns:
            List of episodes, each episode is a list of transitions
        """
        return [self.generate_episode() for _ in range(num_episodes)]