"""Tests for TD learning agents."""

import sys
sys.path.append('.')

import pytest
from mdp.core import State, Action, Transition
from mdp.policy import UniformPolicy
from envs.random_walk import RandomWalkMDP
from agents.td_zero import TD0Agent
from agents.monte_carlo import MonteCarloAgent


def test_td0_agent_initialization():
    """Test TD(0) agent initialization."""
    env = RandomWalkMDP(random_seed=42)
    policy = UniformPolicy(random_seed=42)
    agent = TD0Agent(env, policy, alpha=0.1, gamma=1.0)
    
    # Check initialization
    assert agent.alpha == 0.1
    assert agent.gamma == 1.0
    assert agent.policy == policy
    assert agent.total_updates == 0
    assert agent.cumulative_td_error == 0.0
    
    # Check value function initialization
    assert len(agent.V) > 0
    for state in env.get_states():
        assert agent.get_value(state) == 0.0


def test_td0_agent_update():
    """Test TD(0) single update."""
    env = RandomWalkMDP(random_seed=42)
    policy = UniformPolicy(random_seed=42)
    agent = TD0Agent(env, policy, alpha=0.5, gamma=1.0)
    
    # Create transition: C -> D, reward=0
    state_c = State(3)
    state_d = State(4)
    transition = Transition(state_c, Action("right"), 0.0, state_d, done=False)
    
    # Initial values are 0
    assert agent.get_value(state_c) == 0.0
    assert agent.get_value(state_d) == 0.0
    
    # Update should have no effect since both values are 0 and reward is 0
    td_error_mag = agent.update(transition)
    assert td_error_mag == 0.0
    assert agent.get_value(state_c) == 0.0
    
    # Set value of next state and update again
    agent.set_value(state_d, 0.5)
    td_error_mag = agent.update(transition)
    
    # TD error: δ = 0 + 1.0 * 0.5 - 0.0 = 0.5
    # New value: V(C) = 0.0 + 0.5 * 0.5 = 0.25
    assert agent.get_value(state_c) == 0.25
    assert td_error_mag == 0.5


def test_td0_agent_terminal_update():
    """Test TD(0) update to terminal state."""
    env = RandomWalkMDP(random_seed=42)
    policy = UniformPolicy(random_seed=42)
    agent = TD0Agent(env, policy, alpha=0.5, gamma=1.0)
    
    # Transition to right terminal: E -> RIGHT, reward=1
    state_e = State(5)
    terminal_state = State(6)
    transition = Transition(state_e, Action("right"), 1.0, terminal_state, done=True)
    
    # Initial value is 0
    assert agent.get_value(state_e) == 0.0
    
    # Update with terminal transition
    td_error_mag = agent.update(transition)
    
    # TD error: δ = 1.0 + 0 - 0.0 = 1.0 (terminal state value is 0)
    # New value: V(E) = 0.0 + 0.5 * 1.0 = 0.5
    assert agent.get_value(state_e) == 0.5
    assert td_error_mag == 1.0


def test_monte_carlo_agent_initialization():
    """Test Monte Carlo agent initialization."""
    env = RandomWalkMDP(random_seed=42)
    policy = UniformPolicy(random_seed=42)
    agent = MonteCarloAgent(env, policy, alpha=0.1, gamma=1.0)
    
    # Check initialization
    assert agent.alpha == 0.1
    assert agent.gamma == 1.0
    assert agent.policy == policy
    assert agent.total_episodes == 0
    assert agent.total_returns == 0.0
    
    # Check value function initialization
    assert len(agent.V) > 0
    for state in env.get_states():
        assert agent.get_value(state) == 0.0


def test_monte_carlo_episode_update():
    """Test Monte Carlo update from complete episode."""
    env = RandomWalkMDP(random_seed=42)
    policy = UniformPolicy(random_seed=42)
    agent = MonteCarloAgent(env, policy, alpha=1.0, gamma=1.0)  # α=1 for clear updates
    
    # Simple episode: C -> D -> E -> RIGHT (terminal)
    episode = [
        Transition(State(3), Action("right"), 0.0, State(4), done=False),  # C->D, R=0
        Transition(State(4), Action("right"), 0.0, State(5), done=False),  # D->E, R=0  
        Transition(State(5), Action("right"), 1.0, State(6), done=True)    # E->RIGHT, R=1
    ]
    
    # Update from episode (first visit MC)
    avg_update = agent.update_from_episode(episode)
    
    # Returns: G_0=1, G_1=1, G_2=1
    # Updates: V(C)=1, V(D)=1, V(E)=1
    assert agent.get_value(State(3)) == 1.0  # C
    assert agent.get_value(State(4)) == 1.0  # D
    assert agent.get_value(State(5)) == 1.0  # E
    assert avg_update > 0


def test_monte_carlo_first_visit():
    """Test first-visit Monte Carlo (no repeated state updates)."""
    env = RandomWalkMDP(random_seed=42)
    policy = UniformPolicy(random_seed=42)
    agent = MonteCarloAgent(env, policy, alpha=0.5, gamma=1.0)
    
    # Episode with repeated state: C -> B -> C -> D -> E -> RIGHT (terminal)
    episode = [
        Transition(State(3), Action("left"), 0.0, State(2), done=False),   # C->B, R=0
        Transition(State(2), Action("right"), 0.0, State(3), done=False),  # B->C, R=0
        Transition(State(3), Action("right"), 0.0, State(4), done=False),  # C->D, R=0
        Transition(State(4), Action("right"), 0.0, State(5), done=False),  # D->E, R=0
        Transition(State(5), Action("right"), 1.0, State(6), done=True),   # E->RIGHT, R=1
    ]
    
    # Initial values
    assert agent.get_value(State(3)) == 0.0
    assert agent.get_value(State(2)) == 0.0
    
    # Update - should only update C once (first occurrence)
    agent.update_from_episode(episode)
    
    # Both states should be updated since this is first visit to each
    # Return is 1.0 for all states, so with alpha=0.5, new values should be 0.5
    assert agent.get_value(State(3)) == 0.5  # Updated from first visit (α=0.5, G=1.0)
    assert agent.get_value(State(2)) == 0.5  # Updated from first visit


def test_agent_policy_evaluation():
    """Test agent policy evaluation method."""
    env = RandomWalkMDP(random_seed=42)
    policy = UniformPolicy(random_seed=42)
    
    # Test TD(0)
    td_agent = TD0Agent(env, policy, alpha=0.1, gamma=1.0)
    td_history = td_agent.evaluate_policy(num_episodes=5)
    
    assert len(td_history["episode"]) == 5
    assert len(td_history["total_reward"]) == 5
    assert len(td_history["value_estimates"]) == 5
    
    # Test Monte Carlo  
    mc_agent = MonteCarloAgent(env, policy, alpha=0.1, gamma=1.0)
    mc_history = mc_agent.evaluate_policy(num_episodes=5)
    
    assert len(mc_history["episode"]) == 5
    assert len(mc_history["total_reward"]) == 5
    assert len(mc_history["value_estimates"]) == 5