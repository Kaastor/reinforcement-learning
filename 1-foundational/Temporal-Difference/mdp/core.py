"""
Core MDP components following Sutton & Barto notation.

This module provides the fundamental building blocks for Markov Decision Processes:
- State: Represents states in the state space S
- Action: Represents actions in the action space A  
- Transition: Represents transition dynamics p(s',r|s,a)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Hashable, Optional


@dataclass(frozen=True)
class State:
    """
    Represents a state s ∈ S in the MDP.
    
    Uses frozen dataclass for immutability and hashability.
    The value can be any hashable type (int, str, tuple, etc.).
    """
    value: Hashable
    
    def __str__(self) -> str:
        return f"State({self.value})"


@dataclass(frozen=True) 
class Action:
    """
    Represents an action a ∈ A in the MDP.
    
    Uses frozen dataclass for immutability and hashability.
    The value can be any hashable type (int, str, tuple, etc.).
    """
    value: Hashable
    
    def __str__(self) -> str:
        return f"Action({self.value})"


@dataclass
class Transition:
    """
    Represents a transition (S_t, A_t, R_{t+1}, S_{t+1}) in the MDP.
    
    This captures the core tuple that defines temporal difference learning:
    - state: S_t (current state)
    - action: A_t (action taken) 
    - reward: R_{t+1} (reward received)
    - next_state: S_{t+1} (next state)
    - done: whether episode terminated
    """
    state: State
    action: Action
    reward: float
    next_state: Optional[State]
    done: bool = False
    
    def __str__(self) -> str:
        return f"Transition({self.state} --{self.action}--> {self.next_state}, R={self.reward}, done={self.done})"


class MDP(ABC):
    """
    Abstract base class for Markov Decision Processes.
    
    Defines the interface that all MDP environments must implement,
    following the standard MDP framework from Sutton & Barto.
    """
    
    @abstractmethod
    def get_states(self) -> list[State]:
        """Return all states S in the MDP."""
        pass
    
    @abstractmethod
    def get_actions(self, state: State) -> list[Action]:
        """Return available actions A(s) for a given state s."""
        pass
    
    @abstractmethod
    def step(self, state: State, action: Action) -> Transition:
        """
        Execute action a in state s, return transition.
        
        Returns:
            Transition containing (s, a, r, s', done)
        """
        pass
    
    @abstractmethod
    def reset(self) -> State:
        """Reset environment and return initial state."""
        pass
    
    @abstractmethod
    def is_terminal(self, state: State) -> bool:
        """Check if state is terminal."""
        pass