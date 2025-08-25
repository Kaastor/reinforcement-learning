"""Tests for agent utilities."""

import sys
sys.path.append('.')

import pytest
from mdp.core import State, Action
from agents.utils import EpsilonGreedyPolicy, DecaySchedules, compute_mse


def test_epsilon_greedy_policy():
    """Test ε-greedy action selection."""
    policy = EpsilonGreedyPolicy(epsilon=0.0, random_seed=42)  # Purely greedy
    
    state = State(1)
    actions = [Action("left"), Action("right")]
    q_values = {
        (state, actions[0]): 1.0,  # Better action
        (state, actions[1]): 0.5
    }
    
    # Should always select greedy action when ε=0
    selected = policy.select_action(state, actions, q_values)
    assert selected == actions[0]
    
    # Test with exploration
    policy_explore = EpsilonGreedyPolicy(epsilon=1.0, random_seed=42)  # Purely random
    selected_random = policy_explore.select_action(state, actions, q_values)
    assert selected_random in actions


def test_decay_schedules():
    """Test decay schedule functions."""
    # Linear decay
    linear_decay = DecaySchedules.linear_decay(1.0, 0.1, 100)
    assert linear_decay(0) == pytest.approx(1.0)
    assert linear_decay(100) == pytest.approx(0.1)
    assert linear_decay(50) == pytest.approx(0.55)
    
    # Exponential decay
    exp_decay = DecaySchedules.exponential_decay(1.0, 0.9)
    assert exp_decay(0) == pytest.approx(1.0)
    assert exp_decay(1) == pytest.approx(0.9)
    assert exp_decay(2) == pytest.approx(0.81)
    
    # Step decay
    step_decay = DecaySchedules.step_decay(1.0, 0.5, 10)
    assert step_decay(0) == pytest.approx(1.0)
    assert step_decay(9) == pytest.approx(1.0)
    assert step_decay(10) == pytest.approx(0.5)
    assert step_decay(20) == pytest.approx(0.25)


def test_compute_mse():
    """Test MSE calculation between value functions."""
    s1, s2, s3 = State(1), State(2), State(3)
    
    predicted = {s1: 1.0, s2: 2.0, s3: 3.0}
    target = {s1: 1.1, s2: 1.9, s3: 3.2}
    
    mse = compute_mse(predicted, target)
    expected_mse = ((1.0-1.1)**2 + (2.0-1.9)**2 + (3.0-3.2)**2) / 3
    assert mse == pytest.approx(expected_mse)
    
    # Test with empty dicts
    assert compute_mse({}, {}) == 0.0
    
    # Test with no common states
    assert compute_mse({s1: 1.0}, {s2: 2.0}) == 0.0