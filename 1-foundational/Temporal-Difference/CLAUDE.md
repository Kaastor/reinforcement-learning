# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.


## Project Overview

Goal: build and **understand** core TD ideas by implementing them cleanly (no heavy frameworks), tied directly to the math and to the notation of **Sutton & Barto (2nd ed., 2018)**. By the end you’ll be comfortable with TD(0) for policy evaluation, on-policy control (SARSA), off-policy control (Q-learning) all on tiny, transparent environments.
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


## Project Structure

```
```

## Technical Stack

- **Python version**: Python 3.11
- **Project config**: `pyproject.toml` for configuration and dependency management
- **Environment**: Use virtual environment in `.venv` for dependency isolation
- **Package management**: Use `poetry install` for faster
- **Dependencies**: Separate production and dev dependencies in `pyproject.toml`
- **Project layout**: Standard Python package layout

### Dependencies

[List of deps]

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
