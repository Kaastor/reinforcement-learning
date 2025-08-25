"""Tests for return calculation utilities."""

import sys
sys.path.append('.')

import pytest
from mdp.core import State, Action, Transition  
from td_math.returns import (
    compute_monte_carlo_returns,
    compute_discounted_rewards,
    compute_td_error,
    compute_episode_return
)


def test_monte_carlo_returns():
    """Test Monte Carlo return calculation."""
    # Create simple episode: rewards [1, 2, 3]
    s1, s2, s3, s4 = State(1), State(2), State(3), State(4)
    a = Action("move")
    
    episode = [
        Transition(s1, a, 1.0, s2),
        Transition(s2, a, 2.0, s3), 
        Transition(s3, a, 3.0, s4, done=True)
    ]
    
    # Test with γ=1 (no discounting)
    returns = compute_monte_carlo_returns(episode, gamma=1.0)
    expected = [6.0, 5.0, 3.0]  # [1+2+3, 2+3, 3]
    assert returns == pytest.approx(expected)
    
    # Test with γ=0.5 (discounting)
    returns_discounted = compute_monte_carlo_returns(episode, gamma=0.5)
    expected_discounted = [1 + 0.5*2 + 0.25*3, 2 + 0.5*3, 3.0]
    assert returns_discounted == pytest.approx(expected_discounted)
    
    # Test empty episode
    empty_returns = compute_monte_carlo_returns([])
    assert empty_returns == []


def test_discounted_rewards():
    """Test discounted reward calculation."""
    rewards = [1.0, 2.0, 3.0]
    
    # No discounting
    returns = compute_discounted_rewards(rewards, gamma=1.0)
    assert returns == pytest.approx([6.0, 5.0, 3.0])
    
    # With discounting
    returns_discounted = compute_discounted_rewards(rewards, gamma=0.5)
    expected = [1 + 0.5*2 + 0.25*3, 2 + 0.5*3, 3.0]
    assert returns_discounted == pytest.approx(expected)
    
    # Empty rewards
    empty_returns = compute_discounted_rewards([])
    assert empty_returns == []


def test_td_error():
    """Test TD error calculation."""
    s1, s2 = State(1), State(2)
    a = Action("move")
    
    # Value function
    values = {s1: 5.0, s2: 3.0}
    
    # Non-terminal transition: δ = R + γV(s') - V(s)
    transition = Transition(s1, a, 1.0, s2, done=False)
    delta = compute_td_error(transition, values, gamma=0.9)
    expected = 1.0 + 0.9 * 3.0 - 5.0  # 1 + 2.7 - 5 = -1.3
    assert delta == pytest.approx(expected)
    
    # Terminal transition: δ = R - V(s)
    terminal_transition = Transition(s1, a, 10.0, None, done=True)
    delta_terminal = compute_td_error(terminal_transition, values, gamma=0.9)
    expected_terminal = 10.0 - 5.0  # 10 - 5 = 5
    assert delta_terminal == pytest.approx(expected_terminal)


def test_episode_return():
    """Test total episode return calculation."""
    s1, s2, s3 = State(1), State(2), State(3)
    a = Action("move")
    
    episode = [
        Transition(s1, a, 1.0, s2),
        Transition(s2, a, 2.0, s3, done=True)
    ]
    
    # No discounting
    total_return = compute_episode_return(episode, gamma=1.0)
    assert total_return == pytest.approx(3.0)  # 1 + 2
    
    # With discounting
    discounted_return = compute_episode_return(episode, gamma=0.5)
    expected = 1.0 + 0.5 * 2.0  # 1 + 1 = 2
    assert discounted_return == pytest.approx(expected)
    
    # Empty episode
    empty_return = compute_episode_return([])
    assert empty_return == pytest.approx(0.0)