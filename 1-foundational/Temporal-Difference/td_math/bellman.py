"""
Ground truth value function solver using Bellman equations.

This module implements analytical solutions for value functions v^π and q^π
by solving the Bellman equations: v = r^π + γP^π v.
"""

from typing import Dict

from mdp.core import State, Action, MDP
from mdp.policy import Policy


class BellmanSolver:
    """
    Solves Bellman equations analytically for ground truth value functions.
    
    For policy evaluation: v^π = r^π + γP^π v^π
    Rearranged as: v^π = (I - γP^π)^{-1} r^π
    
    For action-value functions: q^π(s,a) = r(s,a) + γ ∑_{s'} p(s'|s,a) v^π(s')
    """
    
    def __init__(self, mdp: MDP, gamma: float = 1.0):
        """
        Initialize Bellman solver.
        
        Args:
            mdp: The MDP environment
            gamma: Discount factor γ ∈ [0,1]
        """
        self.mdp = mdp
        self.gamma = gamma
        self.states = mdp.get_states()
        self.state_to_idx = {state: i for i, state in enumerate(self.states)}
        self.n_states = len(self.states)
    
    def solve_value_function(self, policy: Policy, max_iterations: int = 1000, tolerance: float = 1e-8) -> Dict[State, float]:
        """
        Solve v^π = (I - γP^π)^{-1} r^π analytically.
        
        For environments where direct matrix inversion is not feasible,
        falls back to iterative policy evaluation.
        
        Args:
            policy: Policy π to evaluate
            max_iterations: Max iterations for iterative fallback
            tolerance: Convergence tolerance
            
        Returns:
            Dictionary mapping states to their values v^π(s)
        """
        # Use iterative method as default for simplicity without numpy
        return self._solve_iterative(policy, max_iterations, tolerance)
    
    def _solve_direct(self, policy: Policy) -> Dict[State, float]:
        """Direct matrix solution: v = (I - γP)^{-1} r (requires numpy)."""
        # This method requires numpy/scipy for matrix operations
        # Falls back to iterative method in current implementation
        return self._solve_iterative(policy, 1000, 1e-8)
    
    def _solve_iterative(self, policy: Policy, max_iterations: int, tolerance: float) -> Dict[State, float]:
        """Iterative policy evaluation: v_{k+1}(s) = ∑_a π(a|s) ∑_{s',r} p(s',r|s,a)[r + γv_k(s')]."""
        # Initialize value function
        V = {state: 0.0 for state in self.states}
        
        for iteration in range(max_iterations):
            V_new = V.copy()
            max_delta = 0.0
            
            for state in self.states:
                if self.mdp.is_terminal(state):
                    V_new[state] = 0.0
                    continue
                
                value = 0.0
                available_actions = self.mdp.get_actions(state)
                
                for action in available_actions:
                    action_prob = policy.get_action_probability(state, action, available_actions)
                    if action_prob > 0:
                        # Sample transition to get expected reward and next state
                        transition = self.mdp.step(state, action)
                        expected_value = transition.reward
                        if transition.next_state is not None and not transition.done:
                            expected_value += self.gamma * V[transition.next_state]
                        value += action_prob * expected_value
                
                V_new[state] = value
                max_delta = max(max_delta, abs(V_new[state] - V[state]))
            
            V = V_new
            if max_delta < tolerance:
                break
        
        return V
    
    def _build_transition_matrix(self, policy: Policy) -> list[list[float]]:
        """Build transition matrix P^π where P^π[i,j] = ∑_a π(a|s_i) p(s_j|s_i,a)."""
        P = [[0.0 for _ in range(self.n_states)] for _ in range(self.n_states)]
        
        for i, state in enumerate(self.states):
            if self.mdp.is_terminal(state):
                P[i][i] = 1.0  # Terminal states transition to themselves
                continue
            
            available_actions = self.mdp.get_actions(state)
            for action in available_actions:
                action_prob = policy.get_action_probability(state, action, available_actions)
                if action_prob > 0:
                    transition = self.mdp.step(state, action)
                    if transition.next_state is not None:
                        j = self.state_to_idx[transition.next_state]
                        P[i][j] += action_prob
        
        return P
    
    def _build_reward_vector(self, policy: Policy) -> list[float]:
        """Build reward vector r^π where r^π[i] = ∑_a π(a|s_i) r(s_i,a)."""
        r = [0.0 for _ in range(self.n_states)]
        
        for i, state in enumerate(self.states):
            if self.mdp.is_terminal(state):
                r[i] = 0.0
                continue
            
            available_actions = self.mdp.get_actions(state)
            for action in available_actions:
                action_prob = policy.get_action_probability(state, action, available_actions)
                if action_prob > 0:
                    transition = self.mdp.step(state, action)
                    r[i] += action_prob * transition.reward
        
        return r
    
    def compute_action_values(self, policy: Policy) -> Dict[tuple[State, Action], float]:
        """
        Compute action-value function q^π(s,a).
        
        q^π(s,a) = r(s,a) + γ ∑_{s'} p(s'|s,a) v^π(s')
        
        Returns:
            Dictionary mapping (state, action) tuples to q-values
        """
        V = self.solve_value_function(policy)
        Q = {}
        
        for state in self.states:
            if self.mdp.is_terminal(state):
                continue
            
            available_actions = self.mdp.get_actions(state)
            for action in available_actions:
                transition = self.mdp.step(state, action)
                q_value = transition.reward
                if transition.next_state is not None and not transition.done:
                    q_value += self.gamma * V[transition.next_state]
                Q[(state, action)] = q_value
        
        return Q