"""
Tests for gridworld environment.
"""

import pytest
from envs.gridworld import GridworldMDP
from mdp.core import State, Action


class TestGridworldMDP:
    """Test gridworld environment implementation."""
    
    def test_initialization(self):
        """Test environment initialization."""
        env = GridworldMDP(grid_size=4, max_steps=100)
        
        assert env.grid_size == 4
        assert env.max_steps == 100
        assert env.start_pos == (0, 0)
        assert env.goal_pos == (3, 3)
        assert env.current_state == env.start_state
        assert len(env.states) == 16  # 4x4 grid
        assert len(env.actions) == 4  # UP, DOWN, LEFT, RIGHT
    
    def test_reset(self):
        """Test environment reset."""
        env = GridworldMDP()
        
        # Move to different state
        env.current_state = State((2, 2))
        env.step_count = 50
        
        # Reset
        reset_state = env.reset()
        
        assert reset_state == env.start_state
        assert env.current_state == env.start_state
        assert env.step_count == 0
    
    def test_terminal_state(self):
        """Test terminal state detection."""
        env = GridworldMDP()
        
        assert not env.is_terminal(env.start_state)
        assert env.is_terminal(env.goal_state)
        assert not env.is_terminal(State((1, 1)))
    
    def test_valid_movements(self):
        """Test valid movements within grid."""
        env = GridworldMDP()
        
        # Start at (0,0), move right to (0,1)
        state = State((0, 0))
        transition = env.step(state, env.RIGHT)
        
        assert transition.next_state == State((0, 1))
        assert transition.reward == -1.0
        assert not transition.done
    
    def test_boundary_collisions(self):
        """Test boundary collision behavior."""
        env = GridworldMDP()
        
        # Try to move up from top row
        state = State((0, 1))
        transition = env.step(state, env.UP)
        
        assert transition.next_state == State((0, 1))  # Stay in place
        assert transition.reward == -1.0
        assert not transition.done
    
    def test_goal_reaching(self):
        """Test reaching goal state."""
        env = GridworldMDP()
        
        # Move to goal
        state = State((2, 3))
        transition = env.step(state, env.DOWN)
        
        assert transition.next_state == env.goal_state
        assert transition.reward == 0.0  # No penalty for reaching goal
        assert transition.done
    
    def test_max_steps_termination(self):
        """Test max steps termination."""
        env = GridworldMDP(max_steps=2)
        
        # Reset and take steps
        env.reset()
        state = State((0, 0))
        
        # First step - should not terminate
        env.step(state, env.RIGHT)
        assert env.step_count == 1
        
        # Second step - should terminate due to max steps
        transition = env.step(State((0, 1)), env.RIGHT)
        assert env.step_count == 2
        assert transition.done
    
    def test_action_space(self):
        """Test action availability."""
        env = GridworldMDP()
        
        # Non-terminal state should have all actions
        actions = env.get_actions(State((1, 1)))
        assert len(actions) == 4
        assert env.UP in actions
        assert env.DOWN in actions
        assert env.LEFT in actions
        assert env.RIGHT in actions
        
        # Terminal state should have no actions
        actions = env.get_actions(env.goal_state)
        assert len(actions) == 0