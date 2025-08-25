"""
Policy evaluation comparison: TD(0) vs Monte Carlo on Random Walk.

This experiment implements Sutton & Barto Example 6.2, comparing
TD(0) and Monte Carlo methods for policy evaluation on the 5-state
random walk environment.

Key comparisons:
- Convergence speed (episodes to reach good estimate)
- Final MSE vs ground truth values
- Sample efficiency (learning per episode)
"""

import logging
from typing import Dict, List, Tuple

from mdp.core import State
from mdp.policy import UniformPolicy
from envs.random_walk import RandomWalkMDP
from agents.td_zero import TD0Agent
from agents.monte_carlo import MonteCarloAgent
from agents.utils import compute_mse, compute_rmse, setup_agent_logging


class PolicyEvaluationExperiment:
    """
    Experiment comparing TD(0) vs Monte Carlo for policy evaluation.
    
    Runs both algorithms on random walk environment with uniform policy
    and tracks convergence to ground truth value function.
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 1.0,
        random_seed: int = 42,
        log_level: int = logging.INFO
    ):
        """
        Initialize policy evaluation experiment.
        
        Args:
            alpha: Learning rate for both algorithms
            gamma: Discount factor  
            random_seed: Random seed for reproducibility
            log_level: Logging level
        """
        self.alpha = alpha
        self.gamma = gamma
        self.random_seed = random_seed
        
        # Set up environment and policy
        self.env = RandomWalkMDP(random_seed=random_seed)
        self.policy = UniformPolicy(random_seed=random_seed)
        
        # Get ground truth values
        self.true_values = self.env.get_true_values()
        
        # Set up logging
        self.logger = setup_agent_logging("PolicyEvaluationExperiment", log_level)
        
        # Initialize agents
        self.td_agent = TD0Agent(
            self.env, self.policy, alpha, gamma, "TD0"
        )
        self.mc_agent = MonteCarloAgent(
            self.env, self.policy, alpha, gamma, "MC"
        )
    
    def run_experiment(
        self,
        num_episodes: int = 100,
        num_runs: int = 1
    ) -> Dict[str, Dict]:
        """
        Run policy evaluation experiment comparing TD(0) vs MC.
        
        Args:
            num_episodes: Episodes per algorithm per run
            num_runs: Number of independent runs for averaging
            
        Returns:
            Dictionary containing results for both algorithms
        """
        self.logger.info(f"Running policy evaluation experiment:")
        self.logger.info(f"  Episodes per run: {num_episodes}")
        self.logger.info(f"  Number of runs: {num_runs}")
        self.logger.info(f"  Learning rate α: {self.alpha}")
        self.logger.info(f"  Discount factor γ: {self.gamma}")
        
        # Store results for both algorithms
        results = {
            "TD0": {"mse_per_episode": [], "rmse_per_episode": [], "final_values": []},
            "MC": {"mse_per_episode": [], "rmse_per_episode": [], "final_values": []}
        }
        
        for run in range(num_runs):
            self.logger.info(f"Starting run {run + 1}/{num_runs}")
            
            # Run TD(0)
            td_mse, td_rmse, td_final = self._run_single_algorithm(
                self.td_agent, "TD(0)", num_episodes, run
            )
            results["TD0"]["mse_per_episode"].append(td_mse)
            results["TD0"]["rmse_per_episode"].append(td_rmse)
            results["TD0"]["final_values"].append(td_final)
            
            # Run Monte Carlo
            mc_mse, mc_rmse, mc_final = self._run_single_algorithm(
                self.mc_agent, "Monte Carlo", num_episodes, run
            )
            results["MC"]["mse_per_episode"].append(mc_mse)
            results["MC"]["rmse_per_episode"].append(mc_rmse)
            results["MC"]["final_values"].append(mc_final)
        
        # Log summary statistics
        self._log_summary(results, num_runs)
        
        return results
    
    def _run_single_algorithm(
        self,
        agent,
        name: str,
        num_episodes: int,
        run: int
    ) -> Tuple[List[float], List[float], Dict[State, float]]:
        """
        Run single algorithm and track MSE vs ground truth.
        
        Returns:
            Tuple of (mse_per_episode, rmse_per_episode, final_values)
        """
        # Reset agent for fresh start
        agent.reset()
        
        mse_history = []
        rmse_history = []
        
        self.logger.debug(f"  Running {name} (run {run + 1})")
        
        # Evaluate policy for specified number of episodes
        training_history = agent.evaluate_policy(num_episodes)
        
        # Calculate MSE after each episode
        for episode_values in training_history["value_estimates"]:
            mse = compute_mse(episode_values, self.true_values)
            rmse = compute_rmse(episode_values, self.true_values)
            mse_history.append(mse)
            rmse_history.append(rmse)
        
        final_values = agent.get_policy_value_estimates()
        
        return mse_history, rmse_history, final_values
    
    def _log_summary(self, results: Dict, num_runs: int) -> None:
        """Log summary statistics for the experiment."""
        self.logger.info("Experiment completed. Summary:")
        
        for alg_name in ["TD0", "MC"]:
            alg_results = results[alg_name]
            
            # Calculate final MSE statistics across runs
            final_mses = [mse_list[-1] for mse_list in alg_results["mse_per_episode"]]
            avg_final_mse = sum(final_mses) / len(final_mses)
            
            self.logger.info(f"  {alg_name}:")
            self.logger.info(f"    Final MSE (avg): {avg_final_mse:.6f}")
            
            if num_runs > 1:
                min_mse = min(final_mses)
                max_mse = max(final_mses)
                self.logger.info(f"    Final MSE (range): [{min_mse:.6f}, {max_mse:.6f}]")
    
    def compare_convergence(
        self,
        results: Dict,
        convergence_threshold: float = 0.01
    ) -> Dict[str, int]:
        """
        Compare convergence speed of algorithms.
        
        Args:
            results: Results from run_experiment
            convergence_threshold: MSE threshold for "convergence"
            
        Returns:
            Dictionary with episodes to convergence for each algorithm
        """
        convergence_episodes = {}
        
        for alg_name in ["TD0", "MC"]:
            episodes_to_converge = []
            
            for mse_history in results[alg_name]["mse_per_episode"]:
                # Find first episode where MSE drops below threshold
                converged_episode = None
                for episode, mse in enumerate(mse_history):
                    if mse <= convergence_threshold:
                        converged_episode = episode + 1
                        break
                
                if converged_episode is not None:
                    episodes_to_converge.append(converged_episode)
                else:
                    episodes_to_converge.append(len(mse_history))  # Never converged
            
            avg_convergence = sum(episodes_to_converge) / len(episodes_to_converge)
            convergence_episodes[alg_name] = avg_convergence
            
            self.logger.info(
                f"{alg_name} average episodes to converge "
                f"(MSE < {convergence_threshold}): {avg_convergence:.1f}"
            )
        
        return convergence_episodes
    
    def display_final_values(self, results: Dict) -> None:
        """Display final learned values vs ground truth."""
        self.logger.info("Final Value Function Comparison:")
        self.logger.info("=" * 50)
        
        # Average final values across runs
        td_final_values = self._average_value_dicts(results["TD0"]["final_values"])
        mc_final_values = self._average_value_dicts(results["MC"]["final_values"])
        
        # Display in table format
        states = [s for s in self.env.nonterminal_states]  # A,B,C,D,E only
        
        print(f"{'State':<8} {'True':<8} {'TD(0)':<8} {'MC':<8} {'TD Err':<8} {'MC Err':<8}")
        print("-" * 50)
        
        for state in states:
            state_name = chr(ord('A') + state.value - 1)
            true_val = self.true_values[state]
            td_val = td_final_values.get(state, 0.0)
            mc_val = mc_final_values.get(state, 0.0)
            td_err = abs(td_val - true_val)
            mc_err = abs(mc_val - true_val)
            
            print(f"{state_name:<8} {true_val:<8.3f} {td_val:<8.3f} {mc_val:<8.3f} {td_err:<8.3f} {mc_err:<8.3f}")
    
    def _average_value_dicts(self, value_dicts: List[Dict[State, float]]) -> Dict[State, float]:
        """Average multiple value dictionaries."""
        if not value_dicts:
            return {}
        
        # Get all states
        all_states = set()
        for val_dict in value_dicts:
            all_states.update(val_dict.keys())
        
        # Average values for each state
        averaged = {}
        for state in all_states:
            values = [val_dict.get(state, 0.0) for val_dict in value_dicts]
            averaged[state] = sum(values) / len(values)
        
        return averaged


def run_basic_experiment():
    """Run basic policy evaluation experiment."""
    experiment = PolicyEvaluationExperiment(
        alpha=0.1,
        gamma=1.0,
        random_seed=42,
        log_level=logging.INFO
    )
    
    # Run experiment
    results = experiment.run_experiment(num_episodes=100, num_runs=1)
    
    # Analyze results
    experiment.compare_convergence(results)
    experiment.display_final_values(results)
    
    return results


if __name__ == "__main__":
    run_basic_experiment()