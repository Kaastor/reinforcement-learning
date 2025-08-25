"""Tests for RL environments."""

import sys
sys.path.append('.')

import pytest
from mdp.core import State, Action
from envs.random_walk import RandomWalkMDP


def test_random_walk_basic():
    """Test basic random walk functionality."""
    env = RandomWalkMDP(random_seed=42)
    
    # Test state structure
    assert len(env.get_states()) == 7  # 0,1,2,3,4,5,6
    assert env.start_state == State(3)  # Start at C
    assert env.is_terminal(State(0))  # Left terminal
    assert env.is_terminal(State(6))  # Right terminal
    assert not env.is_terminal(State(3))  # C is nonterminal
    
    # Test actions
    actions = env.get_actions(State(3))
    assert len(actions) == 2
    assert Action("left") in actions
    assert Action("right") in actions
    
    # Terminal states have no actions
    assert env.get_actions(State(0)) == []
    assert env.get_actions(State(6)) == []


def test_random_walk_transitions():
    """Test random walk transitions."""
    env = RandomWalkMDP(random_seed=42)
    
    # Test movement from state C (3)
    state_c = State(3)
    
    # Move left: C -> B
    transition_left = env.step(state_c, Action("left"))
    assert transition_left.state == state_c
    assert transition_left.action == Action("left")
    assert transition_left.next_state == State(2)  # B
    assert transition_left.reward == 0.0
    assert not transition_left.done
    
    # Move right: C -> D  
    transition_right = env.step(state_c, Action("right"))
    assert transition_right.next_state == State(4)  # D
    assert transition_right.reward == 0.0
    assert not transition_right.done


def test_random_walk_terminal_rewards():
    """Test terminal state rewards."""
    env = RandomWalkMDP(random_seed=42)
    
    # Test reaching right terminal (reward +1)
    state_e = State(5)  # E
    transition = env.step(state_e, Action("right"))
    assert transition.next_state == State(6)  # RIGHT terminal
    assert transition.reward == 1.0
    assert transition.done
    
    # Test reaching left terminal (reward 0)
    state_a = State(1)  # A
    transition = env.step(state_a, Action("left"))
    assert transition.next_state == State(0)  # LEFT terminal
    assert transition.reward == 0.0
    assert transition.done


def test_random_walk_true_values():
    """Test analytical true values."""
    env = RandomWalkMDP()
    true_values = env.get_true_values()
    
    # Check terminal values
    assert true_values[State(0)] == 0.0  # LEFT
    assert true_values[State(6)] == 0.0  # RIGHT
    
    # Check nonterminal values (1/6, 2/6, 3/6, 4/6, 5/6)
    assert true_values[State(1)] == pytest.approx(1/6)  # A
    assert true_values[State(2)] == pytest.approx(2/6)  # B  
    assert true_values[State(3)] == pytest.approx(3/6)  # C
    assert true_values[State(4)] == pytest.approx(4/6)  # D
    assert true_values[State(5)] == pytest.approx(5/6)  # E


def test_random_walk_visualization():
    """Test value function visualization."""
    env = RandomWalkMDP()
    true_values = env.get_true_values()
    
    viz = env.visualize_values(true_values)
    assert "Random Walk Value Function" in viz
    assert "A" in viz and "B" in viz and "C" in viz
    assert "0.167" in viz or "0.333" in viz  # Some true values present