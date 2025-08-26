# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.


## Project Overview

Goal: build and **understand** core TD ideas by implementing them cleanly (no heavy frameworks), tied directly to the math and to the notation of **Sutton & Barto (2nd ed., 2018)**. By the end you’ll be comfortable with TD(0) for policy evaluation, on-policy control (SARSA), off-policy control (Q-learning) all on tiny, transparent environments.

For environment implementation use chosen env from Gymnasium.
---

### Learning outcomes

* Read and write updates in **Sutton & Barto notation**: $S_t, A_t, R_{t+1}, V(s), Q(s,a), \alpha, \gamma, \pi(a\mid s), \delta_t, G_t$.
* Derive and implement core TD updates correctly:

  * **TD(0) policy evaluation**: $ \delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$,
    $V(S_t) \leftarrow V(S_t) + \alpha\,\delta_t$.
  * **SARSA(0)** (on-policy control):
    $\delta_t = R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)$,
    $Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha\,\delta_t$.
  * **Q-learning** (off-policy control):
    $\delta_t = R_{t+1} + \gamma \max_a Q(S_{t+1},a) - Q(S_t,A_t)$,
    $Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha\,\delta_t$.
* Compare TD vs Monte Carlo via **bias-variance** trade-off and **online bootstrapping**.
* Cross-check your results against a **ground-truth value function** computed by solving the Bellman equations.

**Primary theory references:** Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.), Chapter 6 (TD Learning), plus Chapter 3–4 (MDPs, DP/MC background). Keep the book open—your code will cite the exact symbols used there.

### Scope & milestones (8–12 hours total, focused learning)

1. **Foundations (MDP + Notation warm-up)**

   * Short markdown notes: states $\mathcal{S}$, actions $\mathcal{A}$, rewards, transitions $p(s',r\mid s,a)$, returns $G_t$, policies $\pi$.
   * Tiny helper to sample episodes under a fixed policy $\pi$.

2. **Environment 1 — 1D Random Walk (policy evaluation)**

   * Classic **five-state random walk** (nonterminal states $\{A,B,C,D,E\}$, terminals at left/right).
     Actions: Left/Right; $\gamma=1$; reward $0$ until right terminal gives $+1$ (left terminal $0$).
     Use a **fixed policy** (e.g., equiprobable Left/Right).
   * Implement **TD(0)** to estimate $V^\pi$.
   * Compute **ground truth** $v^\pi$ by solving $v = r^\pi + \gamma P^\pi v$ ⇒ $v=(I-\gamma P^\pi)^{-1} r^\pi$.
   * Plot MSE$(V - v^\pi)$ across sweeps; compare **MC** vs **TD(0)**.

3. **Environment 2 — 4×4 Gridworld (control)**

   * Deterministic grid, start at top-left, goal at bottom-right; reward $-1$ per step, $0$ at goal; episode terminates at goal or max steps; $\gamma=1$.
   * Implement **ε-greedy** $\pi$ over $Q$ with linear decay of $\epsilon$.
   * Compare **SARSA(0)** vs **Q-learning** on average return and steps-to-goal over episodes.


## Build & Test Commands

### Using poetry
- Install dependencies: `poetry install`

### Testing
# all tests
`poetry run python -m pytest`

# single test
`poetry run python -m pytest app/tests/test_app.py::test_hello_name -v`


## Current Implementation Status

**Phases Complete**: 3/4 (Foundation + Policy Evaluation + Control)  
**Test Coverage**: 47 passing tests  
**Lines of Code**: ~3000+ lines of clean, well-documented code

## Project Structure

```
td_learning/
├── envs/
│   ├── __init__.py         ✅ Environment package
│   ├── random_walk.py      ✅ 5-state random walk (Sutton & Barto Ex 6.2)
│   └── gridworld.py        ✅ 4x4 deterministic gridworld (Phase 3)
├── agents/
│   ├── __init__.py         ✅ Agent package  
│   ├── base.py            ✅ Abstract agent classes
│   ├── td_zero.py         ✅ TD(0) policy evaluation
│   ├── monte_carlo.py     ✅ MC policy evaluation baseline
│   ├── sarsa.py           ✅ On-policy SARSA(0) control (Phase 3)
│   ├── q_learning.py      ✅ Off-policy Q-learning control (Phase 3)
│   └── utils.py           ✅ ε-greedy, decay schedules
├── mdp/
│   ├── __init__.py         ✅ MDP package
│   ├── core.py            ✅ State, Action, Transition classes
│   └── policy.py          ✅ Policy representation & sampling
├── td_math/
│   ├── __init__.py         ✅ Math utilities package
│   ├── bellman.py         ✅ Ground truth value solvers
│   └── returns.py         ✅ Return calculations (TD error, MC returns)
├── experiments/
│   ├── __init__.py         ✅ Experiment framework package
│   ├── policy_evaluation.py ✅ TD vs MC comparison experiments
│   ├── plotting.py         ✅ ASCII visualization utilities
│   └── control_comparison.py ✅ SARSA vs Q-learning comparison (Phase 3)
├── tests/
│   ├── __init__.py         ✅ Test package
│   ├── test_envs.py        ✅ Random walk environment tests
│   ├── test_agents_td.py   ✅ TD(0) & Monte Carlo agent tests  
│   ├── test_agents_utils.py ✅ Agent utility tests
│   ├── test_experiments.py ✅ Experiment framework tests
│   ├── test_math_returns.py ✅ Mathematical utility tests
│   ├── test_mdp_core.py    ✅ MDP core component tests
│   ├── test_gridworld.py   ✅ Gridworld environment tests (Phase 3)
│   └── test_control_agents.py ✅ SARSA & Q-learning agent tests (Phase 3)
├── demo.py                 ✅ Working demo script (Phase 2)
├── demo_control.py         ✅ Control methods demo script (Phase 3)
└── CLAUDE.md               ✅ Project documentation
```

### Key Achievements

✅ **Phase 1 Complete**: Solid MDP foundation with proper Sutton & Barto notation  
✅ **Phase 2 Complete**: Working TD(0) vs Monte Carlo comparison on Random Walk  
✅ **Phase 3 Complete**: SARSA and Q-learning control methods on 4x4 Gridworld
✅ **Ground Truth Integration**: Analytical solutions for validation  
✅ **Comprehensive Testing**: 47 tests covering all implemented components  
✅ **ASCII Visualization**: No external plotting dependencies needed  
✅ **Control Comparison**: Side-by-side on-policy vs off-policy analysis  

### Current Capabilities

- **Random Walk Environment**: 5-state environment matching Sutton & Barto Example 6.2
- **Gridworld Environment**: 4x4 deterministic grid for control experiments
- **Policy Evaluation**: Both TD(0) and Monte Carlo methods implemented
- **Control Methods**: SARSA(0) on-policy and Q-learning off-policy algorithms
- **Value Function Convergence**: MSE tracking against ground truth values  
- **Experiment Framework**: Automated comparison with configurable parameters
- **Mathematical Foundation**: Proper TD error computation and return calculations
- **Policy Analysis**: Greedy policy extraction and comparison metrics

### Demo Usage

```bash
# Run Phase 2 demo comparing TD(0) vs Monte Carlo on Random Walk
python demo.py

# Run Phase 3 demo comparing SARSA vs Q-learning on Gridworld
python demo_control.py

# Run all tests
poetry run python -m pytest tests/

# Run specific test suites  
PYTHONPATH=. poetry run python -m pytest tests/test_envs.py -v
PYTHONPATH=. poetry run python -m pytest tests/test_agents_td.py -v
PYTHONPATH=. poetry run python -m pytest tests/test_control_agents.py -v
PYTHONPATH=. poetry run python -m pytest tests/test_gridworld.py -v
```

## Technical Stack

- **Python version**: Python 3.11
- **Project config**: `pyproject.toml` for configuration and dependency management
- **Environment**: Use virtual environment in `.venv` for dependency isolation
- **Package management**: Use `poetry install` for faster
- **Dependencies**: Separate production and dev dependencies in `pyproject.toml`
- **Project layout**: Standard Python package layout

### Dependencies

```toml
# Core mathematical operations & RL
numpy = "^1.24.0"              # Efficient array operations, linear algebra
matplotlib = "^3.7.0"          # Plotting TD convergence, MSE comparison  
gymnasium = "^0.29.0"          # Lightweight RL environments
scipy = "^1.11.0"              # Linear system solving for ground truth

# Development & analysis
jupyter = "^1.0.0"             # Interactive development/visualization
pytest = "^8.0.0"              # Testing framework
python-dotenv = "^1.0.0"       # Configuration management
```

## Code Style Guidelines

- **Type hints**: Use native Python type hints (e.g., `list[str]` not `List[str]`)
- **Documentation**: Google-style docstrings for all modules, classes, functions
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Function length**: Keep functions short (< 30 lines) and single-purpose
- **PEP 8**: Follow PEP 8 style guide

## Python Best Practices

- **File handling**: Prefer `pathlib.Path` over `os.path`
- **Debugging**: Use `logging` module instead of `print`
- **Error handling**: Use specific exceptions with context messages and proper logging
- **Data structures**: Use list/dict comprehensions for concise, readable code
- **Function arguments**: Avoid mutable default arguments
- **Data containers**: Leverage `dataclasses` to reduce boilerplate
- **Configuration**: Use environment variables (via `python-dotenv`) for configuration

## Development Patterns & Best Practices

- **Favor simplicity**: Choose the simplest solution that meets requirements
- **DRY principle**: Avoid code duplication; reuse existing functionality
- **Configuration management**: Use environment variables for different environments
- **Focused changes**: Only implement explicitly requested or fully understood changes
- **Preserve patterns**: Follow existing code patterns when fixing bugs
- **File size**: Keep files under 300 lines; refactor when exceeding this limit
- **Test coverage**: Write comprehensive unit and integration tests with `pytest`; include fixtures
- **Test structure**: Use table-driven tests with parameterization for similar test cases
- **Mocking**: Use unittest.mock for external dependencies; don't test implementation details
- **Modular design**: Create reusable, modular components
- **Logging**: Implement appropriate logging levels (debug, info, error)
- **Error handling**: Implement robust error handling for production reliability
- **Security best practices**: Follow input validation and data protection practices
- **Performance**: Optimize critical code sections when necessary


## Core Workflow
- Be sure to typecheck when you’re done making a series of code changes
- Prefer running single tests, and not the whole test suite, for performance

## Implementation Priority
1. Core functionality first (render, state)
2. User interactions
  - Implement only minimal working functionality
3. Minimal unit tests

### Iteration Target
- Around 5 min per cycle
- Keep tests simple, just core functionality checks
- Prioritize working code over perfection for POCs