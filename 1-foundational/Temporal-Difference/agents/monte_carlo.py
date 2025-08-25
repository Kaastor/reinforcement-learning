"""
Monte Carlo policy evaluation agent following Sutton & Barto Algorithm 5.1.

Implements first-visit Monte Carlo for policy evaluation:
- Generate complete episode following policy π  
- For each state S_t in the episode:
  G ← return following first visit to S_t in episode
  V(S_t) ← V(S_t) + α[G - V(S_t)]

This provides an unbiased baseline for comparing with TD(0).
"""

from typing import Dict, Set

from mdp.core import State, Action, Transition, MDP
from mdp.policy import Policy  
from agents.base import ValueBasedAgent
from td_math.returns import compute_monte_carlo_returns


class MonteCarloAgent(ValueBasedAgent):
    """
    First-visit Monte Carlo policy evaluation agent.
    
    Learns state values V(s) ≈ v^π(s) using complete episode returns.
    The update rule is:
        G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...
        V(S_t) ← V(S_t) + α[G_t - V(S_t)]
    """
    
    def __init__(
        self,
        mdp: MDP,
        policy: Policy,
        alpha: float = 0.1,
        gamma: float = 1.0,
        name: str = "MonteCarlo"
    ):
        """
        Initialize Monte Carlo agent.
        
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
        self.total_episodes = 0
        self.total_returns = 0.0
    
    def update(self, transition: Transition) -> float:
        """
        MC doesn't update on single transitions, only complete episodes.
        
        This method is required by base class but not used in MC.
        Use update_from_episode() instead.
        """
        return 0.0
    
    def update_from_episode(self, episode: list[Transition]) -> float:
        """
        Update value function using Monte Carlo returns from complete episode.
        
        Args:
            episode: Complete list of transitions
            
        Returns:
            Average magnitude of updates made
        """
        if not episode:
            return 0.0
        
        # Compute Monte Carlo returns for each step
        returns = compute_monte_carlo_returns(episode, self.gamma)
        
        # Track states we've seen (for first-visit MC)
        visited_states: Set[State] = set()
        total_update = 0.0
        num_updates = 0
        
        # Update values using first-visit Monte Carlo
        for t, (transition, G_t) in enumerate(zip(episode, returns)):
            state = transition.state
            
            # First-visit MC: only update on first occurrence of state
            if state not in visited_states:
                visited_states.add(state)
                
                # MC update: V(S_t) ← V(S_t) + α[G_t - V(S_t)]
                old_value = self.V.get(state, 0.0)
                td_error = G_t - old_value
                new_value = old_value + self.alpha * td_error
                self.V[state] = new_value
                
                total_update += abs(td_error)
                num_updates += 1
                
                self.logger.debug(
                    f"MC Update: S={state.value}, G={G_t:.4f}, "
                    f"V: {old_value:.4f} → {new_value:.4f}"
                )
        
        return total_update / num_updates if num_updates > 0 else 0.0
    
    def evaluate_policy(
        self,
        num_episodes: int,
        max_steps_per_episode: int = 1000
    ) -> Dict[str, list]:
        """
        Evaluate policy using Monte Carlo for multiple episodes.
        
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
            "avg_update": [],
            "value_estimates": []
        }
        
        for episode_num in range(num_episodes):
            # Generate complete episode using policy
            episode = self._generate_episode(max_steps_per_episode)
            
            if not episode:
                self.logger.warning(f"Empty episode {episode_num + 1}")
                continue
            
            # Update value function from complete episode
            avg_update = self.update_from_episode(episode)
            
            # Calculate episode statistics
            episode_reward = sum(t.reward for t in episode)
            episode_steps = len(episode)
            
            # Record episode metrics
            history["episode"].append(episode_num + 1)
            history["total_reward"].append(episode_reward)
            history["steps"].append(episode_steps)
            history["avg_update"].append(avg_update)
            history["value_estimates"].append(self.V.copy())
            
            self.total_episodes += 1
            self.total_returns += episode_reward
            
            self.logger.debug(
                f"Episode {episode_num + 1}: reward={episode_reward:.2f}, "
                f"steps={episode_steps}, avg_update={avg_update:.4f}"
            )
        
        return history
    
    def _generate_episode(self, max_steps: int) -> list[Transition]:
        """
        Generate complete episode following current policy.
        
        Args:
            max_steps: Maximum steps per episode
            
        Returns:
            List of transitions forming complete episode
        """
        episode = []
        state = self.mdp.reset()
        steps = 0
        
        while not self.mdp.is_terminal(state) and steps < max_steps:
            # Select action according to policy
            available_actions = self.mdp.get_actions(state)
            if not available_actions:
                break
                
            action = self.policy.sample_action(state, available_actions)
            
            # Take step in environment
            transition = self.mdp.step(state, action)
            episode.append(transition)
            
            steps += 1
            
            # Move to next state
            state = transition.next_state if transition.next_state else state
            
            if transition.done:
                break
        
        return episode
    
    def get_policy_value_estimates(self) -> Dict[State, float]:
        """Get current value function estimates."""
        return self.V.copy()
    
    def get_average_return(self) -> float:
        """Get average return across all episodes."""
        return (
            self.total_returns / self.total_episodes 
            if self.total_episodes > 0 else 0.0
        )