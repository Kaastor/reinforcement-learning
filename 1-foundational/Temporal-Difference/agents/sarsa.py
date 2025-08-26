"""
SARSA(0) on-policy temporal difference control algorithm.

Implements the SARSA algorithm from Sutton & Barto Chapter 6.4:
- On-policy control using ε-greedy behavior policy
- Action-value function Q(s,a) learning
- TD update: δₜ = Rₜ₊₁ + γQ(Sₜ₊₁,Aₜ₊₁) - Q(Sₜ,Aₜ)
- Update: Q(Sₜ,Aₜ) ← Q(Sₜ,Aₜ) + α δₜ
"""

import random
from typing import Optional, Callable

from mdp.core import State, Action, Transition, MDP
from agents.base import ActionValueBasedAgent
from agents.utils import EpsilonGreedyPolicy


class SarsaAgent(ActionValueBasedAgent):
    """
    SARSA(0) on-policy control agent.
    
    Learns action-value function Q(s,a) using temporal difference updates
    with ε-greedy behavior policy. The key difference from Q-learning is
    that SARSA uses the actual next action A_{t+1} chosen by the policy,
    not the maximum Q-value.
    
    Algorithm (Sutton & Barto Algorithm 6.1):
    1. Initialize Q(s,a) arbitrarily for all s,a
    2. For each episode:
       a. Choose A from S using ε-greedy derived from Q
       b. For each step of episode:
          - Take action A, observe R, S'
          - Choose A' from S' using ε-greedy derived from Q  
          - Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
          - S ← S', A ← A'
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
        Initialize SARSA agent.
        
        Args:
            mdp: The MDP environment
            alpha: Learning rate α ∈ (0,1]
            gamma: Discount factor γ ∈ [0,1]
            epsilon: Exploration probability ε ∈ [0,1]
            epsilon_decay: Optional ε decay schedule
            random_seed: Random seed for reproducibility
        """
        # Set up ε-greedy policy first (needed for reset)
        self.policy = EpsilonGreedyPolicy(epsilon, epsilon_decay, random_seed)
        
        # Track current state-action pair for SARSA updates
        self.current_state: Optional[State] = None
        self.current_action: Optional[Action] = None
        
        if random_seed is not None:
            random.seed(random_seed)
        
        # Initialize parent (which calls reset and initializes Q)
        super().__init__(mdp, alpha, gamma, "SARSA")
    
    def select_action(self, state: State, available_actions: list[Action]) -> Action:
        """
        Select action using ε-greedy policy based on current Q-values.
        
        Args:
            state: Current state
            available_actions: Available actions in this state
            
        Returns:
            Selected action following ε-greedy policy
        """
        return self.policy.select_action(state, available_actions, self.Q)
    
    def update(self, transition: Transition) -> float:
        """
        Update Q-values using SARSA temporal difference rule.
        
        SARSA update: Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
        where A' is the action that will be taken in S' (not max_a Q(S',a))
        
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
            # Get next action using ε-greedy policy (key difference from Q-learning)
            next_actions = self.mdp.get_actions(next_state)
            if next_actions:
                next_action = self.select_action(next_state, next_actions)
                next_q = self.get_q_value(next_state, next_action)
                target = reward + self.gamma * next_q
            else:
                target = reward
        
        # Compute TD error
        td_error = target - current_q
        
        # Update Q-value
        new_q = current_q + self.alpha * td_error
        self.set_q_value(state, action, new_q)
        
        return abs(td_error)
    
    def train_episode_online(self, max_steps: int = 1000) -> dict[str, float]:
        """
        Train agent for one episode using online SARSA updates.
        
        This method implements the full SARSA algorithm where actions
        are selected during the episode and updates happen immediately.
        
        Args:
            max_steps: Maximum steps per episode
            
        Returns:
            Episode metrics (steps, reward, avg_update)
        """
        # Reset environment
        state = self.mdp.reset()
        
        # Select initial action
        available_actions = self.mdp.get_actions(state)
        if not available_actions:
            return {"steps": 0, "total_reward": 0.0, "avg_update": 0.0}
        
        action = self.select_action(state, available_actions)
        
        total_reward = 0.0
        total_update = 0.0
        steps = 0
        
        for _ in range(max_steps):
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
            
            # Select next action (for next iteration)
            next_state = transition.next_state
            next_actions = self.mdp.get_actions(next_state)
            if next_actions:
                next_action = self.select_action(next_state, next_actions)
                state, action = next_state, next_action
            else:
                break
        
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
        self.logger.debug(f"SARSA Episode {self.episode_count}: {metrics}")
        
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
            
            # Find action with maximum Q-value (break ties randomly)
            best_value = float('-inf')
            best_actions = []
            
            for action in actions:
                q_value = self.get_q_value(state, action)
                if q_value > best_value:
                    best_value = q_value
                    best_actions = [action]
                elif q_value == best_value:
                    best_actions.append(action)
            
            greedy_policy[state] = random.choice(best_actions)
        
        return greedy_policy
    
    def reset(self) -> None:
        """Reset agent's Q-values and policy state."""
        super().reset()
        self.policy.reset()
        self.current_state = None
        self.current_action = None
    
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