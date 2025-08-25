"""
TD(0) policy evaluation agent following Sutton & Barto Algorithm 6.1.

Implements temporal difference learning for policy evaluation:
δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)
V(S_t) ← V(S_t) + α δ_t

This is the most basic temporal difference method for estimating v_π.
"""

from typing import Dict

from mdp.core import State, Action, Transition, MDP
from mdp.policy import Policy
from agents.base import ValueBasedAgent
from td_math.returns import compute_td_error


class TD0Agent(ValueBasedAgent):
    """
    TD(0) policy evaluation agent.
    
    Learns state values V(s) ≈ v^π(s) using one-step temporal difference updates.
    The update rule is:
        δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)
        V(S_t) ← V(S_t) + α δ_t
    """
    
    def __init__(
        self,
        mdp: MDP,
        policy: Policy,
        alpha: float = 0.1,
        gamma: float = 1.0,
        name: str = "TD0"
    ):
        """
        Initialize TD(0) agent.
        
        Args:
            mdp: The MDP environment
            policy: Policy π to evaluate
            alpha: Learning rate α ∈ (0,1]
            gamma: Discount factor γ ∈ [0,1]
            name: Agent name for logging
        """
        super().__init__(mdp, alpha, gamma, name)
        self.policy = policy
        
        # Track update statistics
        self.total_updates = 0
        self.cumulative_td_error = 0.0
    
    def update(self, transition: Transition) -> float:
        """
        Update value function using TD(0) rule.
        
        Args:
            transition: (S_t, A_t, R_{t+1}, S_{t+1}, done)
            
        Returns:
            Magnitude of TD error |δ_t|
        """
        # Compute TD error: δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)
        td_error = compute_td_error(transition, self.V, self.gamma)
        
        # Update value: V(S_t) ← V(S_t) + α δ_t
        old_value = self.V.get(transition.state, 0.0)
        new_value = old_value + self.alpha * td_error
        self.V[transition.state] = new_value
        
        # Track statistics
        self.total_updates += 1
        self.cumulative_td_error += abs(td_error)
        
        self.logger.debug(
            f"TD Update: S={transition.state.value}, "
            f"δ={td_error:.4f}, V: {old_value:.4f} → {new_value:.4f}"
        )
        
        return abs(td_error)
    
    def evaluate_policy(
        self,
        num_episodes: int,
        max_steps_per_episode: int = 1000
    ) -> Dict[str, list]:
        """
        Evaluate policy using TD(0) for multiple episodes.
        
        Args:
            num_episodes: Number of episodes to run
            max_steps_per_episode: Maximum steps per episode
            
        Returns:
            Training history with metrics per episode
        """
        history = {
            "episode": [],
            "total_reward": [],
            "steps": [],
            "avg_td_error": [],
            "value_estimates": []
        }
        
        for episode_num in range(num_episodes):
            # Generate episode using current policy
            episode_reward = 0.0
            episode_td_errors = []
            
            state = self.mdp.reset()
            steps = 0
            
            while not self.mdp.is_terminal(state) and steps < max_steps_per_episode:
                # Select action according to policy
                available_actions = self.mdp.get_actions(state)
                if not available_actions:
                    break
                    
                action = self.policy.sample_action(state, available_actions)
                
                # Take step in environment
                transition = self.mdp.step(state, action)
                
                # Update value function using TD(0)
                td_error_magnitude = self.update(transition)
                
                # Track episode statistics
                episode_reward += transition.reward
                episode_td_errors.append(td_error_magnitude)
                steps += 1
                
                # Move to next state
                state = transition.next_state if transition.next_state else state
                
                if transition.done:
                    break
            
            # Record episode metrics
            avg_td_error = sum(episode_td_errors) / len(episode_td_errors) if episode_td_errors else 0.0
            
            history["episode"].append(episode_num + 1)
            history["total_reward"].append(episode_reward)
            history["steps"].append(steps)
            history["avg_td_error"].append(avg_td_error)
            history["value_estimates"].append(self.V.copy())
            
            self.logger.debug(
                f"Episode {episode_num + 1}: reward={episode_reward:.2f}, "
                f"steps={steps}, avg_td_error={avg_td_error:.4f}"
            )
        
        return history
    
    def get_policy_value_estimates(self) -> Dict[State, float]:
        """Get current value function estimates."""
        return self.V.copy()
    
    def get_average_td_error(self) -> float:
        """Get average magnitude of TD errors seen so far."""
        return (
            self.cumulative_td_error / self.total_updates 
            if self.total_updates > 0 else 0.0
        )
    
    def reset_statistics(self) -> None:
        """Reset tracking statistics."""
        self.total_updates = 0
        self.cumulative_td_error = 0.0