"""
Tests for control agents (SARSA and Q-learning).
"""

import pytest
from envs.gridworld import GridworldMDP
from agents.sarsa import SarsaAgent
from agents.q_learning import QLearningAgent
from mdp.core import State, Action, Transition


class TestSarsaAgent:
    """Test SARSA agent implementation."""
    
    def test_initialization(self):
        """Test agent initialization."""
        env = GridworldMDP(grid_size=2)  # Small grid for testing
        agent = SarsaAgent(env, alpha=0.1, gamma=0.9, epsilon=0.1)
        
        assert agent.alpha == 0.1
        assert agent.gamma == 0.9
        assert agent.policy.epsilon == 0.1
        assert agent.name == "SARSA"
        # Q-values should be initialized for all state-action pairs in 2x2 grid
        expected_q_pairs = 3 * 4  # 3 non-terminal states × 4 actions each
        assert len(agent.Q) == expected_q_pairs
    
    def test_q_value_operations(self):
        """Test Q-value getting and setting."""
        env = GridworldMDP(grid_size=2)
        agent = SarsaAgent(env)
        
        state = State((0, 0))
        action = env.UP
        
        # Initial Q-value should be 0
        assert agent.get_q_value(state, action) == 0.0
        
        # Set and get Q-value
        agent.set_q_value(state, action, 1.5)
        assert agent.get_q_value(state, action) == 1.5
    
    def test_action_selection(self):
        """Test action selection with epsilon-greedy."""
        env = GridworldMDP(grid_size=2)
        agent = SarsaAgent(env, epsilon=0.0)  # Greedy policy
        
        state = State((0, 0))
        available_actions = env.get_actions(state)
        
        # Set Q-values to make one action clearly best
        agent.set_q_value(state, env.RIGHT, 2.0)
        agent.set_q_value(state, env.DOWN, 1.0)
        agent.set_q_value(state, env.UP, 0.5)
        agent.set_q_value(state, env.LEFT, 0.5)
        
        # Should select RIGHT action (highest Q-value)
        selected_action = agent.select_action(state, available_actions)
        assert selected_action == env.RIGHT
    
    def test_update_mechanism(self):
        """Test Q-value update from transitions."""
        env = GridworldMDP(grid_size=2)
        agent = SarsaAgent(env, alpha=1.0, gamma=1.0)  # High learning rate
        
        # Create a transition
        state = State((0, 0))
        action = env.RIGHT
        reward = -1.0
        next_state = State((0, 1))
        transition = Transition(state, action, reward, next_state, done=False)
        
        # Update should change Q-value
        initial_q = agent.get_q_value(state, action)
        agent.update(transition)
        updated_q = agent.get_q_value(state, action)
        
        assert updated_q != initial_q
    
    def test_greedy_policy_extraction(self):
        """Test greedy policy extraction."""
        env = GridworldMDP(grid_size=2)
        agent = SarsaAgent(env)
        
        # Set up Q-values for deterministic policy
        state = State((0, 0))
        agent.set_q_value(state, env.RIGHT, 2.0)
        agent.set_q_value(state, env.DOWN, 1.0)
        
        # Extract greedy policy
        policy = agent.get_greedy_policy()
        
        assert state in policy
        assert policy[state] == env.RIGHT


class TestQLearningAgent:
    """Test Q-learning agent implementation."""
    
    def test_initialization(self):
        """Test agent initialization."""
        env = GridworldMDP(grid_size=2)
        agent = QLearningAgent(env, alpha=0.1, gamma=0.9, epsilon=0.1)
        
        assert agent.alpha == 0.1
        assert agent.gamma == 0.9
        assert agent.policy.epsilon == 0.1
        assert agent.name == "Q-Learning"
        # Q-values should be initialized for all state-action pairs in 2x2 grid
        expected_q_pairs = 3 * 4  # 3 non-terminal states × 4 actions each
        assert len(agent.Q) == expected_q_pairs
    
    def test_max_q_value(self):
        """Test maximum Q-value computation."""
        env = GridworldMDP(grid_size=2)
        agent = QLearningAgent(env)
        
        state = State((0, 0))
        
        # Set different Q-values
        agent.set_q_value(state, env.RIGHT, 2.0)
        agent.set_q_value(state, env.DOWN, 1.5)
        agent.set_q_value(state, env.UP, 0.5)
        agent.set_q_value(state, env.LEFT, 1.0)
        
        # Should return maximum
        max_q = agent.get_max_q_value(state)
        assert max_q == 2.0
    
    def test_greedy_action(self):
        """Test greedy action selection."""
        env = GridworldMDP(grid_size=2)
        agent = QLearningAgent(env)
        
        state = State((0, 0))
        available_actions = env.get_actions(state)
        
        # Set Q-values
        agent.set_q_value(state, env.RIGHT, 2.0)
        agent.set_q_value(state, env.DOWN, 1.0)
        agent.set_q_value(state, env.UP, 0.5)
        agent.set_q_value(state, env.LEFT, 0.5)
        
        # Should select greedy action
        greedy_action = agent.get_greedy_action(state, available_actions)
        assert greedy_action == env.RIGHT
    
    def test_update_mechanism(self):
        """Test Q-value update using max target."""
        env = GridworldMDP(grid_size=2)
        agent = QLearningAgent(env, alpha=1.0, gamma=1.0)
        
        # Set up next state Q-values
        next_state = State((0, 1))
        agent.set_q_value(next_state, env.RIGHT, 3.0)
        agent.set_q_value(next_state, env.DOWN, 2.0)
        
        # Create transition
        state = State((0, 0))
        action = env.RIGHT
        reward = -1.0
        transition = Transition(state, action, reward, next_state, done=False)
        
        # Update should use max Q-value from next state
        agent.update(transition)
        updated_q = agent.get_q_value(state, action)
        
        # With alpha=1, gamma=1: Q(s,a) = r + max_a' Q(s',a') = -1 + 3 = 2
        assert abs(updated_q - 2.0) < 1e-6
    
    def test_value_function_extraction(self):
        """Test state value function extraction."""
        env = GridworldMDP(grid_size=2)
        agent = QLearningAgent(env)
        
        state = State((0, 0))
        agent.set_q_value(state, env.RIGHT, 2.0)
        agent.set_q_value(state, env.DOWN, 1.5)
        
        # Extract value function
        values = agent.get_value_function()
        
        assert values[state] == 2.0  # Should be max Q-value