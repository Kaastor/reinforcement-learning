"""
5-state random walk environment following Sutton & Barto Example 6.2.

Classic random walk with states {A, B, C, D, E} and terminal states at both ends.
- Nonterminal states: A, B, C, D, E (indexed 1-5)
- Terminal states: LEFT (0) and RIGHT (6)
- Actions: LEFT, RIGHT
- Transitions: equiprobable left/right movement
- Rewards: 0 everywhere except +1 for reaching right terminal
- Discount: γ = 1.0 (undiscounted episodic task)
"""

import random
from typing import Optional

from mdp.core import State, Action, Transition, MDP


class RandomWalkMDP(MDP):
    """
    5-state random walk MDP.
    
    States: 0(L) - 1(A) - 2(B) - 3(C) - 4(D) - 5(E) - 6(R)
    - States 1-5 are nonterminal (A-E)  
    - States 0,6 are terminal (LEFT, RIGHT)
    - Start state: C (state 3)
    - Actions: LEFT (-1), RIGHT (+1)
    - Reward: +1 for reaching RIGHT terminal, 0 elsewhere
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize random walk environment.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            random.seed(random_seed)
            
        # Define states
        self.LEFT_TERMINAL = State(0)    # Terminal left
        self.RIGHT_TERMINAL = State(6)   # Terminal right
        self.nonterminal_states = [State(i) for i in range(1, 6)]  # A,B,C,D,E
        self.all_states = [self.LEFT_TERMINAL] + self.nonterminal_states + [self.RIGHT_TERMINAL]
        
        # Define actions
        self.LEFT = Action("left")
        self.RIGHT = Action("right") 
        self.actions = [self.LEFT, self.RIGHT]
        
        # Start state (middle state C)
        self.start_state = State(3)  # State C
        self.current_state = self.start_state
    
    def get_states(self) -> list[State]:
        """Return all states including terminals."""
        return self.all_states.copy()
    
    def get_actions(self, state: State) -> list[Action]:
        """Return available actions in given state."""
        if self.is_terminal(state):
            return []
        return self.actions.copy()
    
    def is_terminal(self, state: State) -> bool:
        """Check if state is terminal."""
        return state in [self.LEFT_TERMINAL, self.RIGHT_TERMINAL]
    
    def reset(self) -> State:
        """Reset environment to start state."""
        self.current_state = self.start_state
        return self.current_state
    
    def step(self, state: State, action: Action) -> Transition:
        """
        Execute action in given state and return transition.
        
        Args:
            state: Current state
            action: Action to execute
            
        Returns:
            Transition with next state and reward
        """
        if self.is_terminal(state):
            return Transition(state, action, 0.0, state, done=True)
        
        # Determine next state based on action
        current_pos = state.value
        if action == self.LEFT:
            next_pos = current_pos - 1
        elif action == self.RIGHT:
            next_pos = current_pos + 1
        else:
            raise ValueError(f"Unknown action: {action}")
        
        next_state = State(next_pos)
        
        # Determine reward and done flag
        reward = 1.0 if next_state == self.RIGHT_TERMINAL else 0.0
        done = self.is_terminal(next_state)
        
        # Update current state
        self.current_state = next_state
        
        return Transition(state, action, reward, next_state, done)
    
    def get_transition_probability(self, state: State, action: Action, next_state: State) -> float:
        """
        Get transition probability p(s'|s,a).
        
        In random walk, transitions are deterministic based on action.
        """
        if self.is_terminal(state):
            return 1.0 if next_state == state else 0.0
        
        expected_next_pos = state.value + (1 if action == self.RIGHT else -1)
        return 1.0 if next_state.value == expected_next_pos else 0.0
    
    def get_reward(self, state: State, action: Action, next_state: State) -> float:
        """Get reward for transition."""
        return 1.0 if next_state == self.RIGHT_TERMINAL else 0.0
    
    def get_true_values(self) -> dict[State, float]:
        """
        Return analytical solution for uniform random policy.
        
        Under equiprobable left/right policy, the true values are:
        V(A) = 1/6, V(B) = 2/6, V(C) = 3/6, V(D) = 4/6, V(E) = 5/6
        """
        true_values = {}
        
        # Terminal states
        true_values[self.LEFT_TERMINAL] = 0.0
        true_values[self.RIGHT_TERMINAL] = 0.0
        
        # Nonterminal states (analytical solution for uniform policy)
        for i, state in enumerate(self.nonterminal_states):
            true_values[state] = (i + 1) / 6.0
            
        return true_values
    
    def visualize_values(self, values: dict[State, float]) -> str:
        """
        Create ASCII visualization of value function.
        
        Args:
            values: Dictionary mapping states to values
            
        Returns:
            String representation of values
        """
        lines = []
        lines.append("Random Walk Value Function:")
        lines.append("=" * 40)
        
        # State labels
        state_line = "  L  "
        value_line = " 0.0 "
        
        for state in self.nonterminal_states:
            state_name = chr(ord('A') + state.value - 1)  # Convert 1->A, 2->B, etc.
            state_line += f"  {state_name}  "
            value = values.get(state, 0.0)
            value_line += f"{value:5.3f}"
        
        state_line += "  R  "
        value_line += " 0.0 "
        
        lines.append(state_line)
        lines.append(value_line)
        
        return "\n".join(lines)