"""
Plotting utilities for RL experiment visualization.

Simple ASCII-based plotting for environments without matplotlib.
Provides basic convergence analysis and value function visualization.
"""

from typing import List, Dict, Tuple
from mdp.core import State


def plot_mse_convergence_ascii(
    td_mse: List[float],
    mc_mse: List[float],
    title: str = "MSE Convergence Comparison",
    width: int = 60,
    height: int = 15
) -> str:
    """
    Create ASCII plot of MSE convergence for TD(0) vs Monte Carlo.
    
    Args:
        td_mse: TD(0) MSE values per episode
        mc_mse: Monte Carlo MSE values per episode
        title: Plot title
        width: Plot width in characters
        height: Plot height in characters
        
    Returns:
        Multi-line string containing ASCII plot
    """
    if not td_mse and not mc_mse:
        return "No data to plot"
    
    # Combine data and find ranges
    all_mse = td_mse + mc_mse
    max_mse = max(all_mse) if all_mse else 1.0
    min_mse = min(all_mse) if all_mse else 0.0
    max_episodes = max(len(td_mse), len(mc_mse))
    
    # Add some padding to ranges
    mse_range = max_mse - min_mse
    if mse_range == 0:
        mse_range = 1.0
    max_mse += mse_range * 0.1
    min_mse -= mse_range * 0.1
    min_mse = max(0, min_mse)  # MSE can't be negative
    
    # Create plot grid
    plot_lines = []
    
    # Title
    plot_lines.append(title.center(width))
    plot_lines.append("=" * width)
    
    # Y-axis labels and plot area
    for row in range(height):
        y_pos = height - 1 - row  # Flip for correct orientation
        y_value = min_mse + (max_mse - min_mse) * (y_pos / (height - 1))
        
        # Y-axis label
        line = f"{y_value:6.3f}|"
        
        # Plot data points
        for col in range(width - 8):  # Leave space for y-axis
            x_pos = col / (width - 9)  # Normalize to [0,1]
            episode = int(x_pos * (max_episodes - 1))
            
            char = " "
            
            # Check TD data
            if episode < len(td_mse):
                td_y_pos = (td_mse[episode] - min_mse) / (max_mse - min_mse)
                td_y_row = int(td_y_pos * (height - 1))
                if abs(td_y_row - y_pos) <= 0.5:
                    char = "T"
            
            # Check MC data (overrides TD if both present)
            if episode < len(mc_mse):
                mc_y_pos = (mc_mse[episode] - min_mse) / (max_mse - min_mse)
                mc_y_row = int(mc_y_pos * (height - 1))
                if abs(mc_y_row - y_pos) <= 0.5:
                    char = "M" if char != "T" else "*"  # * for overlap
            
            line += char
        
        plot_lines.append(line)
    
    # X-axis
    x_axis = " " * 7 + "+" + "-" * (width - 8)
    plot_lines.append(x_axis)
    
    # X-axis labels
    x_labels = " " * 7
    for i in range(0, max_episodes, max(1, max_episodes // 8)):
        pos = int((i / max_episodes) * (width - 8))
        if pos < len(x_labels):
            x_labels += f"{i}".ljust(8)[:8]
        else:
            break
    plot_lines.append(x_labels)
    
    # Legend
    plot_lines.append("")
    plot_lines.append("Legend: T=TD(0), M=Monte Carlo, *=Both")
    
    return "\n".join(plot_lines)


def create_value_comparison_table(
    true_values: Dict[State, float],
    td_values: Dict[State, float],
    mc_values: Dict[State, float],
    state_names: List[str] = None
) -> str:
    """
    Create table comparing learned vs true values.
    
    Args:
        true_values: Ground truth values
        td_values: TD(0) learned values
        mc_values: Monte Carlo learned values
        state_names: Optional state names (defaults to A,B,C,...)
        
    Returns:
        Formatted table string
    """
    # Get sorted states
    states = sorted(true_values.keys(), key=lambda s: s.value)
    
    if state_names is None:
        state_names = [chr(ord('A') + i) for i in range(len(states))]
    
    lines = []
    lines.append("Value Function Comparison")
    lines.append("=" * 60)
    lines.append(f"{'State':<8} {'True':<10} {'TD(0)':<10} {'MC':<10} {'TD Err':<10} {'MC Err':<10}")
    lines.append("-" * 60)
    
    for i, state in enumerate(states):
        state_name = state_names[i] if i < len(state_names) else str(state.value)
        true_val = true_values.get(state, 0.0)
        td_val = td_values.get(state, 0.0)
        mc_val = mc_values.get(state, 0.0)
        td_err = abs(td_val - true_val)
        mc_err = abs(mc_val - true_val)
        
        lines.append(
            f"{state_name:<8} {true_val:<10.4f} {td_val:<10.4f} {mc_val:<10.4f} "
            f"{td_err:<10.4f} {mc_err:<10.4f}"
        )
    
    # Summary statistics
    all_td_errors = [abs(td_values.get(s, 0.0) - true_values.get(s, 0.0)) for s in states]
    all_mc_errors = [abs(mc_values.get(s, 0.0) - true_values.get(s, 0.0)) for s in states]
    
    avg_td_error = sum(all_td_errors) / len(all_td_errors) if all_td_errors else 0.0
    avg_mc_error = sum(all_mc_errors) / len(all_mc_errors) if all_mc_errors else 0.0
    
    lines.append("-" * 60)
    lines.append(f"{'Average':<8} {'':<10} {'':<10} {'':<10} {avg_td_error:<10.4f} {avg_mc_error:<10.4f}")
    
    return "\n".join(lines)


def create_learning_summary(
    td_history: Dict,
    mc_history: Dict,
    convergence_threshold: float = 0.01
) -> str:
    """
    Create summary of learning performance.
    
    Args:
        td_history: TD(0) training history
        mc_history: Monte Carlo training history
        convergence_threshold: MSE threshold for convergence
        
    Returns:
        Summary string
    """
    lines = []
    lines.append("Learning Performance Summary")
    lines.append("=" * 40)
    
    # Episodes to convergence
    def episodes_to_converge(mse_history: List[float]) -> int:
        for i, mse in enumerate(mse_history):
            if mse <= convergence_threshold:
                return i + 1
        return len(mse_history)  # Never converged
    
    if "avg_td_error" in td_history and "avg_update" in mc_history:
        td_mse = [compute_mse_from_values(vals, true_vals) 
                 for vals, true_vals in zip(td_history["value_estimates"], td_history.get("true_values", []))]
        mc_mse = [compute_mse_from_values(vals, true_vals)
                 for vals, true_vals in zip(mc_history["value_estimates"], mc_history.get("true_values", []))]
        
        if td_mse:
            td_converge = episodes_to_converge(td_mse)
            lines.append(f"TD(0) episodes to converge: {td_converge}")
        
        if mc_mse:
            mc_converge = episodes_to_converge(mc_mse)
            lines.append(f"Monte Carlo episodes to converge: {mc_converge}")
    
    # Final performance
    if td_history.get("total_reward"):
        avg_td_reward = sum(td_history["total_reward"]) / len(td_history["total_reward"])
        lines.append(f"TD(0) average reward: {avg_td_reward:.3f}")
    
    if mc_history.get("total_reward"):
        avg_mc_reward = sum(mc_history["total_reward"]) / len(mc_history["total_reward"])
        lines.append(f"Monte Carlo average reward: {avg_mc_reward:.3f}")
    
    return "\n".join(lines)


def compute_mse_from_values(predicted: Dict[State, float], target: Dict[State, float]) -> float:
    """Helper to compute MSE between value dictionaries."""
    if not predicted or not target:
        return 0.0
    
    common_states = set(predicted.keys()) & set(target.keys())
    if not common_states:
        return 0.0
    
    squared_errors = []
    for state in common_states:
        error = predicted[state] - target[state]
        squared_errors.append(error ** 2)
    
    return sum(squared_errors) / len(squared_errors)