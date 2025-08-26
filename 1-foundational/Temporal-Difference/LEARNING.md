# Learning Path: Temporal Difference Methods

## Overview
This project provides a hands-on approach to understanding temporal difference (TD) learning through clean implementations tied directly to Sutton & Barto notation. The goal is deep conceptual understanding, not just working code.

## Prerequisites
- Linear algebra basics (matrix operations, system of equations)
- Probability fundamentals (expectation, conditional probability)
- Basic Python programming (classes, numpy arrays)
- Familiarity with Markov Decision Processes (states, actions, rewards, policies)

## Learning Path

### Phase 1: Foundation Building (2-3 hours)
**Goal**: Understand the mathematical framework and notation

1. **Read Sutton & Barto Chapter 3-4** 
   - Focus on MDP formulation: $\mathcal{S}, \mathcal{A}, p(s',r|s,a), \pi(a|s)$
   - Understand value functions: $V^\pi(s), Q^\pi(s,a)$
   - Grasp the Bellman equations

2. **Explore the MDP Foundation**
   - Read `mdp/core.py` - understand State, Action, Transition classes
   - Read `mdp/policy.py` - see how policies are represented and sampled
   - Run simple examples to see state transitions

3. **Understand Ground Truth Computation**
   - Study `td_math/bellman.py` 
   - See how $v^\pi = (I - \gamma P^\pi)^{-1} r^\pi$ is implemented
   - This will be your benchmark for all TD methods

### Phase 2: Policy Evaluation Deep Dive (3-4 hours)
**Goal**: Master the difference between Monte Carlo and TD methods

4. **Study Random Walk Environment**
   - Read `envs/random_walk.py` - understand the 5-state setup
   - This matches Sutton & Barto Example 6.2 exactly
   - Note: $\gamma = 1$, reward structure, terminal states

5. **Implement and Compare Monte Carlo vs TD(0)**
   - Study `agents/monte_carlo.py` - see how full returns $G_t$ are used
   - Study `agents/td_zero.py` - focus on the TD error: $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$
   - Run `demo.py` to see convergence patterns

6. **Analyze the Bias-Variance Trade-off**
   - MC: unbiased but high variance (uses full episode)
   - TD(0): biased but lower variance (bootstraps from current estimate)
   - Plot MSE convergence - TD should converge faster

7. **Mathematical Deep Dive**
   - Derive why $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ is the TD error
   - Understand why $V(S_t) \leftarrow V(S_t) + \alpha \delta_t$ moves toward true value
   - Study learning rate schedules and their impact

### Phase 3: Control Methods Mastery (3-4 hours)
**Goal**: Understand on-policy vs off-policy control

8. **Study Gridworld Environment**
   - Read `envs/gridworld.py` - deterministic 4×4 grid
   - Understand state representation, action space, reward structure
   - Note the episode termination conditions

9. **Master µ-Greedy Policy**
   - Study `agents/utils.py` - see epsilon decay schedules
   - Understand exploration-exploitation trade-off
   - See how policies are derived from Q-values

10. **Compare SARSA vs Q-Learning**
    - Study `agents/sarsa.py`: $\delta_t = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)$
    - Study `agents/q_learning.py`: $\delta_t = R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)$
    - Key difference: SARSA uses actual next action, Q-learning uses optimal next action

11. **Run Control Experiments**
    - Execute `demo_control.py` 
    - Compare learning curves, final policies, convergence rates
    - Understand why Q-learning can be more aggressive (off-policy)

### Phase 4: Advanced Understanding (2-3 hours)
**Goal**: Connect theory to implementation details

12. **Study Experimental Framework**
    - Read `experiments/policy_evaluation.py` and `experiments/control_comparison.py`
    - Understand how experiments are structured for fair comparison
    - See how hyperparameters affect learning

13. **Analyze Convergence Properties**
    - Why does TD(0) converge faster than MC in policy evaluation?
    - When might SARSA be preferred over Q-learning?
    - How do learning rates and exploration schedules affect performance?

14. **Implementation Deep Dive**
    - Study how episodes are generated and processed
    - Understand state representation choices
    - See how updates are applied in practice

15. **Run Your Own Experiments**
    - Modify hyperparameters (±, ³, µ schedules)
    - Try different initial value functions
    - Compare different exploration strategies

## Key Learning Checkpoints

**After Phase 1**: Can you write the Bellman equation for a simple MDP?
**After Phase 2**: Can you explain why TD(0) converges faster than MC?
**After Phase 3**: Can you predict when SARSA vs Q-learning would perform differently?
**After Phase 4**: Can you implement a new TD variant from scratch?

## Hands-On Exercises

1. **Modify the Random Walk**: Change reward structure or add stochasticity
2. **Create New Gridworld**: Try 5×5 with obstacles or stochastic transitions
3. **Implement TD(»)**: Extend the codebase with eligibility traces
4. **Compare Learning Rates**: Systematically study ± impact on convergence
5. **Policy Analysis**: Extract and visualize learned policies

## Common Pitfalls to Avoid

- **Don't skip the math**: Each line of code implements a specific equation
- **Don't ignore ground truth**: Always validate against analytical solutions
- **Don't rush to control**: Master policy evaluation first
- **Don't neglect hyperparameters**: They dramatically affect learning
- **Don't confuse algorithms**: SARSA and Q-learning differ by one crucial step

## Testing Your Understanding

Run the test suite (`poetry run python -m pytest`) and understand what each test verifies:
- Environment dynamics match expected behavior
- Agents implement correct update equations  
- Mathematical utilities produce accurate results
- Experiments generate expected convergence patterns

## Going Deeper

After mastering this codebase:
- Read Sutton & Barto Chapters 7-12 (n-step methods, planning)
- Implement function approximation (neural networks for value functions)
- Study policy gradient methods
- Explore modern deep RL algorithms (DQN, A3C, PPO)

## Success Metrics

You've truly learned from this project when you can:
1. Implement any TD method from its mathematical description
2. Predict comparative performance before running experiments
3. Debug convergence issues by understanding the underlying theory
4. Extend the framework to new environments and algorithms
5. Teach these concepts to others using the codebase as reference