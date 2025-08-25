"""
Return calculation utilities following Sutton & Barto notation.

This module implements different types of returns used in RL:
- Monte Carlo returns: G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...
- n-step returns: G_{t:t+n} = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γ^n V(S_{t+n})
- λ-returns for eligibility traces
"""

from typing import List

from mdp.core import Transition


def compute_monte_carlo_returns(episode: List[Transition], gamma: float = 1.0) -> List[float]:
    """
    Compute Monte Carlo returns G_t for each step in an episode.
    
    G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... + γ^{T-t-1}R_T
    
    Args:
        episode: List of transitions in the episode
        gamma: Discount factor γ ∈ [0,1]
        
    Returns:
        List of returns G_t for each time step t
    """
    if not episode:
        return []
    
    returns = []
    G = 0.0  # Return, computed backwards from the end
    
    # Compute returns backwards: G_t = R_{t+1} + γG_{t+1}
    for transition in reversed(episode):
        G = transition.reward + gamma * G
        returns.append(G)
    
    # Reverse to get forward order
    returns.reverse()
    return returns


def compute_discounted_rewards(rewards: List[float], gamma: float = 1.0) -> List[float]:
    """
    Compute discounted cumulative rewards from a sequence of rewards.
    
    Args:
        rewards: Sequence of rewards [R_1, R_2, ..., R_T]
        gamma: Discount factor γ ∈ [0,1]
        
    Returns:
        Discounted returns for each time step
    """
    if not rewards:
        return []
    
    returns = []
    G = 0.0
    
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.append(G)
    
    returns.reverse()
    return returns


def compute_n_step_returns(
    episode: List[Transition], 
    values: dict, 
    n: int, 
    gamma: float = 1.0
) -> List[float]:
    """
    Compute n-step returns G_{t:t+n}.
    
    G_{t:t+n} = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γ^n V(S_{t+n})
    
    Args:
        episode: List of transitions
        values: Value function estimates V(s)
        n: Number of steps for n-step return
        gamma: Discount factor
        
    Returns:
        List of n-step returns
    """
    if not episode:
        return []
    
    returns = []
    T = len(episode)
    
    for t in range(T):
        G = 0.0
        
        # Sum rewards for min(n, T-t) steps
        for k in range(min(n, T - t)):
            G += (gamma ** k) * episode[t + k].reward
        
        # Add bootstrapped value if episode doesn't end within n steps
        if t + n < T and episode[t + n - 1].next_state is not None:
            next_state = episode[t + n - 1].next_state
            if next_state in values:
                G += (gamma ** n) * values[next_state]
        
        returns.append(G)
    
    return returns


def compute_td_error(
    transition: Transition, 
    values: dict, 
    gamma: float = 1.0
) -> float:
    """
    Compute TD error δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t).
    
    This is the core temporal difference error used in TD learning algorithms.
    
    Args:
        transition: Single transition (S_t, A_t, R_{t+1}, S_{t+1})
        values: Current value function estimates
        gamma: Discount factor
        
    Returns:
        TD error δ_t
    """
    current_value = values.get(transition.state, 0.0)
    
    if transition.done or transition.next_state is None:
        # Terminal transition: δ_t = R_{t+1} - V(S_t)
        return transition.reward - current_value
    else:
        # Non-terminal: δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)
        next_value = values.get(transition.next_state, 0.0)
        return transition.reward + gamma * next_value - current_value


def compute_gae_returns(
    episode: List[Transition],
    values: dict,
    gamma: float = 1.0,
    lambda_: float = 1.0
) -> tuple[List[float], List[float]]:
    """
    Compute Generalized Advantage Estimation (GAE) returns and advantages.
    
    GAE(λ): A_t^{GAE(λ)} = ∑_{l=0}^{∞} (γλ)^l δ_{t+l}
    
    Args:
        episode: List of transitions
        values: Value function estimates
        gamma: Discount factor
        lambda_: GAE parameter λ ∈ [0,1]
        
    Returns:
        Tuple of (returns, advantages)
    """
    if not episode:
        return [], []
    
    # Compute TD errors
    td_errors = []
    for transition in episode:
        delta = compute_td_error(transition, values, gamma)
        td_errors.append(delta)
    
    # Compute GAE advantages backwards
    advantages = []
    gae = 0.0
    
    for delta in reversed(td_errors):
        gae = delta + gamma * lambda_ * gae
        advantages.append(gae)
    
    advantages.reverse()
    
    # Returns are advantages plus baseline values
    returns = []
    for i, transition in enumerate(episode):
        baseline_value = values.get(transition.state, 0.0)
        returns.append(advantages[i] + baseline_value)
    
    return returns, advantages


def compute_episode_return(episode: List[Transition], gamma: float = 1.0) -> float:
    """
    Compute total discounted return for an episode.
    
    G_0 = ∑_{t=0}^{T-1} γ^t R_{t+1}
    
    Args:
        episode: List of transitions in episode
        gamma: Discount factor
        
    Returns:
        Total discounted return
    """
    total_return = 0.0
    for t, transition in enumerate(episode):
        total_return += (gamma ** t) * transition.reward
    return total_return