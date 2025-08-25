"""Tests for experiment framework."""

import sys
sys.path.append('.')

import pytest
from experiments.policy_evaluation import PolicyEvaluationExperiment
from experiments.plotting import plot_mse_convergence_ascii, create_value_comparison_table
from mdp.core import State


def test_policy_evaluation_experiment_init():
    """Test policy evaluation experiment initialization."""
    experiment = PolicyEvaluationExperiment(
        alpha=0.1, 
        gamma=1.0, 
        random_seed=42
    )
    
    # Check components are initialized
    assert experiment.alpha == 0.1
    assert experiment.gamma == 1.0
    assert experiment.random_seed == 42
    assert experiment.env is not None
    assert experiment.policy is not None
    assert experiment.true_values is not None
    assert experiment.td_agent is not None  
    assert experiment.mc_agent is not None


def test_policy_evaluation_experiment_run():
    """Test running policy evaluation experiment."""
    experiment = PolicyEvaluationExperiment(
        alpha=0.2,  # Higher learning rate for faster convergence in test
        gamma=1.0,
        random_seed=42
    )
    
    # Run short experiment
    results = experiment.run_experiment(num_episodes=10, num_runs=1)
    
    # Check results structure
    assert "TD0" in results
    assert "MC" in results
    assert "mse_per_episode" in results["TD0"]
    assert "rmse_per_episode" in results["TD0"]
    assert "final_values" in results["TD0"]
    assert len(results["TD0"]["mse_per_episode"]) == 1  # One run
    assert len(results["TD0"]["mse_per_episode"][0]) == 10  # Ten episodes


def test_convergence_analysis():
    """Test convergence analysis methods."""
    experiment = PolicyEvaluationExperiment(alpha=0.1, gamma=1.0, random_seed=42)
    
    # Mock results with simple MSE decay
    mock_results = {
        "TD0": {
            "mse_per_episode": [[0.1, 0.05, 0.02, 0.008]],  # Converges at episode 4
        },
        "MC": {
            "mse_per_episode": [[0.2, 0.1, 0.05, 0.015]],   # Converges at episode 4
        }
    }
    
    convergence = experiment.compare_convergence(mock_results, convergence_threshold=0.01)
    
    # Both should converge at episode 4
    assert convergence["TD0"] == 4.0
    assert convergence["MC"] == 4.0


def test_ascii_plotting():
    """Test ASCII plotting utilities."""
    # Simple test data
    td_mse = [0.1, 0.08, 0.05, 0.03, 0.01]
    mc_mse = [0.2, 0.15, 0.1, 0.05, 0.02]
    
    plot = plot_mse_convergence_ascii(td_mse, mc_mse, width=40, height=8)
    
    # Check basic structure
    assert "MSE Convergence" in plot
    assert "T=" in plot  # Legend for TD
    assert "M=" in plot  # Legend for MC
    assert len(plot.split('\n')) >= 8  # At least height lines


def test_value_comparison_table():
    """Test value comparison table creation."""
    # Sample values
    true_values = {State(1): 0.2, State(2): 0.4, State(3): 0.6}
    td_values = {State(1): 0.18, State(2): 0.42, State(3): 0.58}
    mc_values = {State(1): 0.22, State(2): 0.38, State(3): 0.62}
    
    table = create_value_comparison_table(
        true_values, td_values, mc_values, 
        state_names=["A", "B", "C"]
    )
    
    # Check table structure
    assert "Value Function Comparison" in table
    assert "A" in table and "B" in table and "C" in table
    assert "0.200" in table  # True value present
    assert "Average" in table  # Summary row present