#!/usr/bin/env python3
"""
Demo script for Phase 3 control methods (SARSA vs Q-learning).

This script demonstrates the comparison between on-policy (SARSA) and
off-policy (Q-learning) temporal difference control methods on a 4x4 gridworld.
"""

from experiments.control_comparison import ControlComparison


def main():
    """Run control comparison demo."""
    print("=" * 70)
    print("TEMPORAL DIFFERENCE CONTROL METHODS COMPARISON")
    print("Phase 3: SARSA (on-policy) vs Q-learning (off-policy)")
    print("=" * 70)
    
    # Set up experiment with smaller parameters for quick demo
    comparison = ControlComparison(
        grid_size=4,
        max_steps_per_episode=50,
        alpha=0.1,
        gamma=0.95,
        epsilon_initial=0.3,
        epsilon_final=0.01,
        epsilon_decay_episodes=200,
        random_seed=42
    )
    
    # Run comparison with fewer episodes for demo
    results = comparison.run_comparison(
        num_episodes=300,
        eval_interval=100,
        verbose=True
    )
    
    print("\n" + "=" * 70)
    print("LEARNING CURVES")
    print("=" * 70)
    
    # Show learning curves
    curves = comparison.plot_learning_curves()
    print(curves)
    
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    # Show summary statistics
    stats = comparison.get_summary_statistics()
    
    print(f"Final Performance (last 100 episodes):")
    print(f"  SARSA:      {stats['sarsa_final_reward_mean']:6.2f} ± {stats['sarsa_final_reward_std']:4.2f}")
    print(f"  Q-learning: {stats['qlearning_final_reward_mean']:6.2f} ± {stats['qlearning_final_reward_std']:4.2f}")
    print(f"  Policy Agreement: {stats['policy_agreement']:.1%}")
    
    print(f"\nSteps to Goal (average):")
    print(f"  SARSA:      {stats['sarsa_final_steps_mean']:5.1f}")
    print(f"  Q-learning: {stats['qlearning_final_steps_mean']:5.1f}")
    
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print("• SARSA (on-policy) learns the policy it's following (ε-greedy)")
    print("• Q-learning (off-policy) learns the optimal policy regardless of behavior")
    print("• Both should converge to similar performance in this environment")
    print("• Differences arise in exploration vs exploitation trade-offs")
    print("=" * 70)


if __name__ == "__main__":
    main()