"""
Q-learning off-policy temporal difference control algorithm.

Implements the Q-learning algorithm from Sutton & Barto Chapter 6.5:
- Off-policy control using ε-greedy behavior policy
- Action-value function Q(s,a) learning  
- TD update: δₜ = Rₜ₊₁ + γ max_a Q(Sₜ₊₁,a) - Q(Sₜ,Aₜ)
- Update: Q(Sₜ,Aₜ) ← Q(Sₜ,Aₜ) + α δₜ
"""

import random
from typing import Optional, Callable

from mdp.core import State, Action, Transition, MDP
from agents.base import ActionValueBasedAgent
from agents.utils import EpsilonGreedyPolicy


class QLearningAgent(ActionValueBasedAgent):
    """
    Q-learning off-policy control agent.
    
    Learns action-value function Q(s,a) using temporal difference updates.
    The key difference from SARSA is that Q-learning uses the maximum
    Q-value for the next state (greedy action) regardless of the actual
    action taken by the behavior policy.
    
    Algorithm (Sutton & Barto Algorithm 6.2):
    1. Initialize Q(s,a) arbitrarily for all s,a  
    2. For each episode:
       a. Initialize S
       b. For each step of episode:
          - Choose A from S using ε-greedy derived from Q
          - Take action A, observe R, S'
          - Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]
          - S ← S'
    """
    
    def __init__(
        self,
        mdp: MDP,
        alpha: float = 0.1,
        gamma: float = 1.0,
        epsilon: float = 0.1,
        epsilon_decay: Optional[Callable[[int], float]] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize Q-learning agent.
        
        Args:
            mdp: The MDP environment
            alpha: Learning rate α ∈ (0,1]
            gamma: Discount factor γ ∈ [0,1]
            epsilon: Exploration probability ε ∈ [0,1]
            epsilon_decay: Optional ε decay schedule  
            random_seed: Random seed for reproducibility
        """
        # Set up ε-greedy policy for behavior (needed for reset)
        self.policy = EpsilonGreedyPolicy(epsilon, epsilon_decay, random_seed)
        
        if random_seed is not None:
            random.seed(random_seed)
        
        # Initialize parent (which calls reset and initializes Q)
        super().__init__(mdp, alpha, gamma, "Q-Learning")
    
    def select_action(self, state: State, available_actions: list[Action]) -> Action:
        """
        Select action using ε-greedy behavior policy.
        
        Args:
            state: Current state
            available_actions: Available actions in this state
            
        Returns:
            Selected action following ε-greedy policy
        """
        return self.policy.select_action(state, available_actions, self.Q)
    
    def get_max_q_value(self, state: State) -> float:
        """
        Get maximum Q-value for a state: max_a Q(s,a).
        
        Args:
            state: State to evaluate
            
        Returns:
            Maximum Q-value across all actions in state
        """
        if self.mdp.is_terminal(state):
            return 0.0
        
        actions = self.mdp.get_actions(state)
        if not actions:
            return 0.0
        
        q_values = [self.get_q_value(state, action) for action in actions]
        return max(q_values)
    
    def get_greedy_action(self, state: State, available_actions: list[Action]) -> Action:
        """
        Get greedy action: argmax_a Q(s,a).
        
        Args:
            state: Current state
            available_actions: Available actions in this state
            
        Returns:
            Greedy action (breaks ties randomly)
        """
        if not available_actions:
            raise ValueError("No available actions")
        
        best_value = float('-inf')
        best_actions = []
        
        for action in available_actions:
            q_value = self.get_q_value(state, action)
            if q_value > best_value:
                best_value = q_value
                best_actions = [action]
            elif q_value == best_value:
                best_actions.append(action)
        
        return random.choice(best_actions)
    
    def update(self, transition: Transition) -> float:
        """
        Update Q-values using Q-learning temporal difference rule.
        
        Q-learning update: Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]
        where max_a Q(S',a) is the maximum Q-value (greedy action value)
        
        Args:
            transition: (S_t, A_t, R_{t+1}, S_{t+1}, done)
            
        Returns:
            Magnitude of TD error |δ_t|
        """
        state = transition.state
        action = transition.action
        reward = transition.reward
        next_state = transition.next_state
        done = transition.done
        
        # Current Q-value
        current_q = self.get_q_value(state, action)
        
        # Compute target value
        if done:
            # Terminal state: no future rewards
            target = reward
        else:
            # Use maximum Q-value for next state (key difference from SARSA)
            max_next_q = self.get_max_q_value(next_state)
            target = reward + self.gamma * max_next_q
        
        # Compute TD error
        td_error = target - current_q
        
        # Update Q-value
        new_q = current_q + self.alpha * td_error
        self.set_q_value(state, action, new_q)
        
        return abs(td_error)
    
    def train_episode_online(self, max_steps: int = 1000) -> dict[str, float]:
        """
        Train agent for one episode using online Q-learning updates.
        
        This method implements the full Q-learning algorithm where actions
        are selected during the episode and updates happen immediately.
        
        Args:
            max_steps: Maximum steps per episode
            
        Returns:
            Episode metrics (steps, reward, avg_update)
        """
        # Reset environment
        state = self.mdp.reset()
        
        total_reward = 0.0
        total_update = 0.0
        steps = 0
        
        for _ in range(max_steps):
            # Select action using ε-greedy policy
            available_actions = self.mdp.get_actions(state)
            if not available_actions:
                break
            
            action = self.select_action(state, available_actions)
            
            # Take action and observe result
            transition = self.mdp.step(state, action)
            
            # Update Q-values
            td_error_magnitude = self.update(transition)
            
            # Track metrics
            total_reward += transition.reward
            total_update += td_error_magnitude
            steps += 1
            
            # Check if episode ended
            if transition.done:
                break
            
            # Move to next state
            state = transition.next_state
        
        # Update episode count and epsilon
        self.episode_count += 1
        self.policy.update_epsilon(self.episode_count)
        
        # Record metrics
        metrics = {
            "episode": self.episode_count,
            "steps": steps,
            "total_reward": total_reward,
            "avg_update": total_update / steps if steps > 0 else 0.0
        }
        
        self.training_history.append(metrics)
        self.logger.debug(f"Q-Learning Episode {self.episode_count}: {metrics}")
        
        return metrics
    
    def get_greedy_policy(self) -> dict[State, Action]:
        """
        Extract the greedy policy π*(s) = argmax_a Q(s,a).
        
        Returns:
            Dictionary mapping states to greedy actions
        """
        greedy_policy = {}
        
        for state in self.mdp.get_states():
            if self.mdp.is_terminal(state):
                continue
                
            actions = self.mdp.get_actions(state)
            if not actions:
                continue
            
            greedy_policy[state] = self.get_greedy_action(state, actions)
        
        return greedy_policy
    
    def get_value_function(self) -> dict[State, float]:
        """
        Extract state value function V(s) = max_a Q(s,a).
        
        Returns:
            Dictionary mapping states to their values
        """
        value_function = {}
        
        for state in self.mdp.get_states():
            value_function[state] = self.get_max_q_value(state)
        
        return value_function
    
    def reset(self) -> None:
        """Reset agent's Q-values and policy state."""
        super().reset()
        self.policy.reset()
    
    def save_checkpoint(self) -> dict:
        """Save agent state including policy parameters."""
        checkpoint = super().save_checkpoint()
        checkpoint.update({
            "epsilon": self.policy.epsilon,
            "episode_count": self.policy.episode_count,
            "Q": self.Q.copy()
        })
        return checkpoint
    
    def load_checkpoint(self, checkpoint: dict) -> None:
        """Load agent state including policy parameters."""
        super().load_checkpoint(checkpoint)
        self.policy.epsilon = checkpoint.get("epsilon", self.policy.initial_epsilon)
        self.policy.episode_count = checkpoint.get("episode_count", 0)
        if "Q" in checkpoint:
            self.Q = checkpoint["Q"].copy()