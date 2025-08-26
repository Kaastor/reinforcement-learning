"""
Control comparison experiments: SARSA vs Q-learning on gridworld.

This module implements experiments to compare on-policy (SARSA) and 
off-policy (Q-learning) temporal difference control methods following
Sutton & Barto Chapter 6.5-6.6.
"""

import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

from envs.gridworld import GridworldMDP
from agents.sarsa import SarsaAgent
from agents.q_learning import QLearningAgent
from agents.utils import DecaySchedules
from experiments.plotting import plot_mse_convergence_ascii


@dataclass
class ComparisonResults:
    """Results from control algorithm comparison."""
    sarsa_rewards: List[float]
    qlearning_rewards: List[float]
    sarsa_steps: List[int]
    qlearning_steps: List[int]
    sarsa_values: Dict
    qlearning_values: Dict
    sarsa_policy: Dict
    qlearning_policy: Dict


class ControlComparison:
    """
    Compare SARSA and Q-learning on gridworld control task.
    
    This experiment demonstrates the difference between on-policy (SARSA)
    and off-policy (Q-learning) methods, particularly in how they handle
    exploration during learning.
    """
    
    def __init__(
        self,
        grid_size: int = 4,
        max_steps_per_episode: int = 100,
        alpha: float = 0.1,
        gamma: float = 1.0,
        epsilon_initial: float = 0.1,
        epsilon_final: float = 0.01,
        epsilon_decay_episodes: int = 500,
        random_seed: Optional[int] = None
    ):
        """
        Initialize control comparison experiment.
        
        Args:
            grid_size: Size of square gridworld
            max_steps_per_episode: Maximum steps per episode
            alpha: Learning rate for both algorithms
            gamma: Discount factor
            epsilon_initial: Initial exploration rate
            epsilon_final: Final exploration rate
            epsilon_decay_episodes: Episodes for epsilon decay
            random_seed: Random seed for reproducibility
        """
        self.grid_size = grid_size
        self.max_steps = max_steps_per_episode
        self.alpha = alpha
        self.gamma = gamma
        self.random_seed = random_seed
        
        # Set up logging
        self.logger = logging.getLogger("experiments.control_comparison")
        
        # Create environment
        self.env = GridworldMDP(grid_size, max_steps_per_episode)
        
        # Set up epsilon decay schedule
        epsilon_decay = DecaySchedules.linear_decay(
            epsilon_initial, epsilon_final, epsilon_decay_episodes
        )
        
        # Create agents
        self.sarsa_agent = SarsaAgent(
            self.env, alpha, gamma, epsilon_initial, epsilon_decay, random_seed
        )
        
        self.qlearning_agent = QLearningAgent(
            self.env, alpha, gamma, epsilon_initial, epsilon_decay, random_seed
        )
        
        # Results storage
        self.results: Optional[ComparisonResults] = None
    
    def run_comparison(
        self,
        num_episodes: int = 1000,
        eval_interval: int = 50,
        verbose: bool = True
    ) -> ComparisonResults:
        """
        Run comparison between SARSA and Q-learning.
        
        Args:
            num_episodes: Number of training episodes per algorithm
            eval_interval: Episodes between progress logging
            verbose: Whether to print progress
            
        Returns:
            ComparisonResults with training metrics and final policies
        """
        if verbose:
            print(f"Running control comparison: SARSA vs Q-learning")
            print(f"Environment: {self.grid_size}x{self.grid_size} gridworld")
            print(f"Episodes: {num_episodes}, α={self.alpha}, γ={self.gamma}")
            print("=" * 60)
        
        # Train SARSA
        if verbose:
            print("Training SARSA (on-policy)...")
        
        sarsa_rewards, sarsa_steps = self._train_agent(
            self.sarsa_agent, num_episodes, eval_interval, "SARSA", verbose
        )
        
        # Train Q-learning  
        if verbose:
            print("Training Q-learning (off-policy)...")
        
        qlearning_rewards, qlearning_steps = self._train_agent(
            self.qlearning_agent, num_episodes, eval_interval, "Q-learning", verbose
        )
        
        # Extract final policies and value functions
        sarsa_policy = self.sarsa_agent.get_greedy_policy()
        qlearning_policy = self.qlearning_agent.get_greedy_policy()
        
        sarsa_values = self._extract_state_values(self.sarsa_agent)
        qlearning_values = self._extract_state_values(self.qlearning_agent)
        
        # Store results
        self.results = ComparisonResults(
            sarsa_rewards=sarsa_rewards,
            qlearning_rewards=qlearning_rewards,
            sarsa_steps=sarsa_steps,
            qlearning_steps=qlearning_steps,
            sarsa_values=sarsa_values,
            qlearning_values=qlearning_values,
            sarsa_policy=sarsa_policy,
            qlearning_policy=qlearning_policy
        )
        
        if verbose:
            self._print_final_comparison()
        
        return self.results
    
    def _train_agent(
        self,
        agent,
        num_episodes: int,
        eval_interval: int,
        agent_name: str,
        verbose: bool
    ) -> Tuple[List[float], List[int]]:
        """Train a single agent and track performance."""
        rewards = []
        steps = []
        
        for episode in range(num_episodes):
            # Train one episode
            metrics = agent.train_episode_online(self.max_steps)
            
            rewards.append(metrics["total_reward"])
            steps.append(metrics["steps"])
            
            # Print progress
            if verbose and (episode + 1) % eval_interval == 0:
                recent_rewards = rewards[-eval_interval:]
                recent_steps = steps[-eval_interval:]
                avg_reward = sum(recent_rewards) / len(recent_rewards)
                avg_steps = sum(recent_steps) / len(recent_steps)
                
                print(f"  Episode {episode + 1:4d}: "
                      f"avg_reward={avg_reward:6.2f}, avg_steps={avg_steps:5.1f}, "
                      f"ε={agent.policy.epsilon:.3f}")
        
        return rewards, steps
    
    def _extract_state_values(self, agent) -> Dict:
        """Extract state value function V(s) = max_a Q(s,a) from agent."""
        values = {}
        for state in self.env.get_states():
            values[state] = agent.get_value(state)
        return values
    
    def _print_final_comparison(self) -> None:
        """Print final comparison results."""
        if not self.results:
            return
        
        print("\n" + "=" * 60)
        print("FINAL COMPARISON RESULTS")
        print("=" * 60)
        
        # Performance summary
        sarsa_final_reward = sum(self.results.sarsa_rewards[-100:]) / 100
        qlearn_final_reward = sum(self.results.qlearning_rewards[-100:]) / 100
        sarsa_final_steps = sum(self.results.sarsa_steps[-100:]) / 100
        qlearn_final_steps = sum(self.results.qlearning_steps[-100:]) / 100
        
        print(f"Average performance (last 100 episodes):")
        print(f"  SARSA:      reward={sarsa_final_reward:6.2f}, steps={sarsa_final_steps:5.1f}")
        print(f"  Q-learning: reward={qlearn_final_reward:6.2f}, steps={qlearn_final_steps:5.1f}")
        
        # Policy comparison
        print("\nFinal Policies:")
        print("\nSARSA Policy:")
        print(self.env.visualize_policy(lambda s: self.results.sarsa_policy.get(s)))
        
        print("\nQ-learning Policy:")
        print(self.env.visualize_policy(lambda s: self.results.qlearning_policy.get(s)))
        
        # Check policy agreement
        agreement = self._compute_policy_agreement()
        print(f"\nPolicy Agreement: {agreement:.1%} of states have same greedy action")
    
    def _compute_policy_agreement(self) -> float:
        """Compute fraction of states where SARSA and Q-learning agree on greedy action."""
        if not self.results:
            return 0.0
        
        total_states = 0
        agreed_states = 0
        
        for state in self.env.get_states():
            if self.env.is_terminal(state):
                continue
                
            sarsa_action = self.results.sarsa_policy.get(state)
            qlearn_action = self.results.qlearning_policy.get(state)
            
            if sarsa_action is not None and qlearn_action is not None:
                total_states += 1
                if sarsa_action == qlearn_action:
                    agreed_states += 1
        
        return agreed_states / total_states if total_states > 0 else 0.0
    
    def plot_learning_curves(self) -> str:
        """
        Create ASCII plot of learning curves.
        
        Returns:
            ASCII plot string showing reward curves for both algorithms
        """
        if not self.results:
            return "No results available. Run comparison first."
        
        # Smooth rewards with moving average
        window_size = 50
        sarsa_smooth = self._moving_average(self.results.sarsa_rewards, window_size)
        qlearn_smooth = self._moving_average(self.results.qlearning_rewards, window_size)
        
        # Use the existing plotting function (treat as "MSE" for plotting purposes)
        return plot_mse_convergence_ascii(
            sarsa_smooth, 
            qlearn_smooth, 
            title="Learning Curves: SARSA vs Q-learning (Smoothed Rewards)",
            width=60,
            height=15
        )
    
    def _moving_average(self, data: List[float], window_size: int) -> List[float]:
        """Compute moving average for smoothing."""
        if len(data) < window_size:
            return data
        
        smoothed = []
        for i in range(window_size - 1, len(data)):
            window = data[i - window_size + 1:i + 1]
            smoothed.append(sum(window) / len(window))
        
        return smoothed
    
    def get_summary_statistics(self) -> Dict[str, float]:
        """
        Get summary statistics for both algorithms.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.results:
            return {}
        
        # Final performance (last 100 episodes)
        sarsa_final = self.results.sarsa_rewards[-100:]
        qlearn_final = self.results.qlearning_rewards[-100:]
        
        sarsa_steps_final = self.results.sarsa_steps[-100:]
        qlearn_steps_final = self.results.qlearning_steps[-100:]
        
        return {
            "sarsa_final_reward_mean": sum(sarsa_final) / len(sarsa_final),
            "sarsa_final_reward_std": self._compute_std(sarsa_final),
            "sarsa_final_steps_mean": sum(sarsa_steps_final) / len(sarsa_steps_final),
            "qlearning_final_reward_mean": sum(qlearn_final) / len(qlearn_final),
            "qlearning_final_reward_std": self._compute_std(qlearn_final),
            "qlearning_final_steps_mean": sum(qlearn_steps_final) / len(qlearn_steps_final),
            "policy_agreement": self._compute_policy_agreement()
        }
    
    def _compute_std(self, data: List[float]) -> float:
        """Compute standard deviation."""
        if len(data) < 2:
            return 0.0
        
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
        return variance ** 0.5