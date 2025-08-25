"""Tests for MDP core components."""

import sys
sys.path.append('.')

import pytest
from mdp.core import State, Action, Transition, MDP
from mdp.policy import UniformPolicy, DeterministicPolicy, EpisodeGenerator


def test_state_creation():
    """Test State class basic functionality."""
    s1 = State(1)
    s2 = State("A")
    s3 = State((0, 1))
    
    assert s1.value == 1
    assert s2.value == "A"
    assert s3.value == (0, 1)
    
    # Test immutability and hashability
    assert hash(s1) == hash(State(1))
    assert s1 == State(1)


def test_action_creation():
    """Test Action class basic functionality."""
    a1 = Action("left")
    a2 = Action(0)
    
    assert a1.value == "left"
    assert a2.value == 0
    
    # Test immutability and hashability
    assert hash(a1) == hash(Action("left"))
    assert a1 == Action("left")


def test_transition_creation():
    """Test Transition class functionality."""
    s1 = State(1)
    s2 = State(2)
    a = Action("move")
    
    t1 = Transition(s1, a, 1.0, s2, False)
    t2 = Transition(s1, a, 0.0, None, True)  # Terminal transition
    
    assert t1.state == s1
    assert t1.action == a
    assert t1.reward == 1.0
    assert t1.next_state == s2
    assert t1.done is False
    
    assert t2.done is True
    assert t2.next_state is None


def test_uniform_policy():
    """Test UniformPolicy action selection."""
    policy = UniformPolicy(random_seed=42)
    state = State(1)
    actions = [Action("left"), Action("right"), Action("up")]
    
    # Test probability calculation
    for action in actions:
        prob = policy.get_action_probability(state, action, actions)
        assert prob == pytest.approx(1.0 / 3.0)
    
    # Test invalid action
    invalid_action = Action("invalid")
    prob = policy.get_action_probability(state, invalid_action, actions)
    assert prob == 0.0
    
    # Test sampling (should return valid action)
    selected_action = policy.sample_action(state, actions)
    assert selected_action in actions


def test_deterministic_policy():
    """Test DeterministicPolicy functionality."""
    state1 = State(1)
    state2 = State(2)
    action1 = Action("left")
    action2 = Action("right")
    
    policy_map = {state1: action1, state2: action2}
    policy = DeterministicPolicy(policy_map)
    
    # Test action selection
    available_actions = [action1, action2]
    selected = policy.sample_action(state1, available_actions)
    assert selected == action1
    
    # Test probabilities
    prob1 = policy.get_action_probability(state1, action1, available_actions)
    prob2 = policy.get_action_probability(state1, action2, available_actions)
    assert prob1 == 1.0
    assert prob2 == 0.0