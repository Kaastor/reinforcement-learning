"""
4x4 gridworld environment for control experiments.

Deterministic gridworld with:
- 4x4 grid of states (16 total)  
- Start state: top-left corner (0,0)
- Goal state: bottom-right corner (3,3)
- Actions: UP, DOWN, LEFT, RIGHT
- Reward: -1 per step, 0 at goal
- Episode terminates at goal or max steps
- Discount: γ = 1.0 (undiscounted episodic task)
"""

from typing import Optional, Tuple
from mdp.core import State, Action, Transition, MDP


class GridworldMDP(MDP):
    """
    4x4 deterministic gridworld MDP.
    
    Grid layout (row, col):
    (0,0) (0,1) (0,2) (0,3)
    (1,0) (1,1) (1,2) (1,3)
    (2,0) (2,1) (2,2) (2,3)
    (3,0) (3,1) (3,2) (3,3)
    
    - Start: (0,0) top-left
    - Goal: (3,3) bottom-right
    - Reward: -1 per step, 0 at goal
    """
    
    def __init__(self, grid_size: int = 4, max_steps: int = 100):
        """
        Initialize gridworld environment.
        
        Args:
            grid_size: Size of square grid (default 4x4)
            max_steps: Maximum steps per episode
        """
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.step_count = 0
        
        # Define states as (row, col) tuples
        self.start_pos = (0, 0)
        self.goal_pos = (grid_size - 1, grid_size - 1)
        
        # Create all valid states
        self.states = []
        for row in range(grid_size):
            for col in range(grid_size):
                self.states.append(State((row, col)))
        
        self.start_state = State(self.start_pos)
        self.goal_state = State(self.goal_pos)
        self.current_state = self.start_state
        
        # Define actions
        self.UP = Action("up")
        self.DOWN = Action("down")
        self.LEFT = Action("left")
        self.RIGHT = Action("right")
        self.actions = [self.UP, self.DOWN, self.LEFT, self.RIGHT]
        
        # Action deltas
        self.action_deltas = {
            self.UP: (-1, 0),
            self.DOWN: (1, 0),
            self.LEFT: (0, -1),
            self.RIGHT: (0, 1)
        }
    
    def get_states(self) -> list[State]:
        """Return all states."""
        return self.states.copy()
    
    def get_actions(self, state: State) -> list[Action]:
        """Return available actions in given state."""
        if self.is_terminal(state):
            return []
        return self.actions.copy()
    
    def is_terminal(self, state: State) -> bool:
        """Check if state is terminal (goal reached)."""
        return state == self.goal_state
    
    def reset(self) -> State:
        """Reset environment to start state."""
        self.current_state = self.start_state
        self.step_count = 0
        return self.current_state
    
    def step(self, state: State, action: Action) -> Transition:
        """
        Execute action in given state and return transition.
        
        Args:
            state: Current state (row, col)
            action: Action to execute
            
        Returns:
            Transition with next state and reward
        """
        self.step_count += 1
        
        # If already at goal, stay there with 0 reward
        if self.is_terminal(state):
            return Transition(state, action, 0.0, state, done=True)
        
        # If max steps reached, end episode
        if self.step_count >= self.max_steps:
            return Transition(state, action, -1.0, state, done=True)
        
        # Calculate next position
        current_row, current_col = state.value
        delta_row, delta_col = self.action_deltas[action]
        next_row = current_row + delta_row
        next_col = current_col + delta_col
        
        # Check bounds and stay in place if out of bounds
        if (next_row < 0 or next_row >= self.grid_size or 
            next_col < 0 or next_col >= self.grid_size):
            next_row, next_col = current_row, current_col
        
        next_state = State((next_row, next_col))
        
        # Determine reward and done flag
        if next_state == self.goal_state:
            reward = 0.0  # Goal reached, no penalty
            done = True
        else:
            reward = -1.0  # Step penalty
            done = False
        
        # Update current state
        self.current_state = next_state
        
        return Transition(state, action, reward, next_state, done)
    
    def get_transition_probability(self, state: State, action: Action, next_state: State) -> float:
        """
        Get transition probability p(s'|s,a).
        
        Gridworld has deterministic transitions.
        """
        if self.is_terminal(state):
            return 1.0 if next_state == state else 0.0
        
        # Calculate expected next state
        current_row, current_col = state.value
        delta_row, delta_col = self.action_deltas[action]
        next_row = current_row + delta_row
        next_col = current_col + delta_col
        
        # Check bounds
        if (next_row < 0 or next_row >= self.grid_size or 
            next_col < 0 or next_col >= self.grid_size):
            next_row, next_col = current_row, current_col
        
        expected_next_state = State((next_row, next_col))
        return 1.0 if next_state == expected_next_state else 0.0
    
    def get_reward(self, state: State, action: Action, next_state: State) -> float:
        """Get reward for transition."""
        if next_state == self.goal_state:
            return 0.0
        return -1.0
    
    def visualize_policy(self, policy_fn) -> str:
        """
        Create ASCII visualization of policy.
        
        Args:
            policy_fn: Function that takes state and returns action
            
        Returns:
            String representation of policy
        """
        lines = []
        lines.append("Gridworld Policy:")
        lines.append("=" * (self.grid_size * 4 + 1))
        
        action_symbols = {
            self.UP: "↑",
            self.DOWN: "↓", 
            self.LEFT: "←",
            self.RIGHT: "→"
        }
        
        for row in range(self.grid_size):
            line = "|"
            for col in range(self.grid_size):
                state = State((row, col))
                if state == self.goal_state:
                    symbol = "G"
                elif state == self.start_state:
                    symbol = "S"
                else:
                    action = policy_fn(state)
                    symbol = action_symbols.get(action, "?")
                line += f" {symbol} |"
            lines.append(line)
            lines.append("=" * (self.grid_size * 4 + 1))
        
        return "\n".join(lines)
    
    def visualize_values(self, values: dict[State, float]) -> str:
        """
        Create ASCII visualization of value function.
        
        Args:
            values: Dictionary mapping states to values
            
        Returns:
            String representation of values
        """
        lines = []
        lines.append("Gridworld Value Function:")
        lines.append("=" * (self.grid_size * 8 + 1))
        
        for row in range(self.grid_size):
            line = "|"
            for col in range(self.grid_size):
                state = State((row, col))
                value = values.get(state, 0.0)
                line += f"{value:6.2f} |"
            lines.append(line)
            lines.append("=" * (self.grid_size * 8 + 1))
        
        return "\n".join(lines)
    
    def pos_to_state_index(self, pos: Tuple[int, int]) -> int:
        """Convert (row, col) position to flat state index."""
        row, col = pos
        return row * self.grid_size + col
    
    def state_index_to_pos(self, index: int) -> Tuple[int, int]:
        """Convert flat state index to (row, col) position."""
        row = index // self.grid_size
        col = index % self.grid_size
        return (row, col)