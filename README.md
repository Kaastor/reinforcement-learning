# Road to Reinforcement Learning

#### 1. **Foundational (Tabular and Early RL Methods)**
These establish core concepts like temporal difference learning and policy gradients, applicable in discrete environments and as theoretical foundations.

| Method | Why Cornerstone? | Seminal Paper Title | Authors | Year | Link |
|--------|------------------|---------------------|---------|------|------|
| Temporal Difference (TD) Learning | Basis for value prediction without full models; enables bootstrapping for efficient learning in sequential decisions. | Learning to Predict by the Methods of Temporal Differences | Richard S. Sutton | 1988 | https://link.springer.com/article/10.1007/BF00115009 |
| Q-Learning | Off-policy value-based method for optimal action selection; foundational for model-free RL in unknown environments. | Q-Learning | Christopher J.C.H. Watkins and Peter Dayan | 1992 | https://link.springer.com/article/10.1007/BF00992698 |
| REINFORCE (Policy Gradient) | Simple gradient-based policy optimization; core for direct policy search without value functions. | Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning | Ronald J. Williams | 1992 | https://link.springer.com/article/10.1007/BF00992696 |
| Policy Iteration / Value Iteration | Dynamic programming for exact solutions in known MDPs; essential for understanding planning and convergence guarantees. | Policy Gradient Methods for Reinforcement Learning with Function Approximation | Richard S. Sutton et al. | 1999 | https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf |
| Dyna (Integrated Planning and Learning) | Hybrid model-based/model-free; introduces imagination for sample efficiency using learned models. | Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming | Richard S. Sutton | 1990 | https://proceedings.mlr.press/v0/sutton90a/sutton90a.pdf |

#### 2. **Value-Based Deep RL Methods**
These extend tabular methods to high-dimensional spaces using neural networks, crucial for scaling to real-world tasks like games.

| Method | Why Cornerstone? | Seminal Paper Title | Authors | Year | Link |
|--------|------------------|---------------------|---------|------|------|
| Deep Q-Network (DQN) | First successful deep RL for Atari; introduces experience replay and target networks for stability. | Playing Atari with Deep Reinforcement Learning | Volodymyr Mnih et al. | 2013 | https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf |
| Double DQN | Addresses overestimation bias in Q-learning; improves stability in value-based methods. | Deep Reinforcement Learning with Double Q-Learning | Hado van Hasselt et al. | 2015 | https://arxiv.org/abs/1509.06461 |
| Dueling DQN | Separates value and advantage streams; enhances generalization in action-rich environments. | Dueling Network Architectures for Deep Reinforcement Learning | Ziyu Wang et al. | 2015 | https://arxiv.org/abs/1511.06581 |
| Prioritized Experience Replay (PER) | Weights replay samples by TD error; boosts efficiency by focusing on informative experiences. | Prioritized Experience Replay | Tom Schaul et al. | 2015 | https://arxiv.org/abs/1511.05952 |
| Rainbow DQN | Combines multiple DQN improvements (e.g., double, dueling, PER); benchmark for integrated value-based RL. | Rainbow: Combining Improvements in Deep Reinforcement Learning | Matteo Hessel et al. | 2017 | https://arxiv.org/abs/1710.02298 |

#### 3. **Policy-Based and Actor-Critic Methods**
These focus on stochastic policies for continuous actions; actor-critic hybrids combine value estimation with policy gradients for variance reduction.

| Method | Why Cornerstone? | Seminal Paper Title | Authors | Year | Link |
|--------|------------------|---------------------|---------|------|------|
| Asynchronous Advantage Actor-Critic (A3C) | Parallel actors for faster training; introduces advantage estimation for better credit assignment. | Asynchronous Methods for Deep Reinforcement Learning | Volodymyr Mnih et al. | 2016 | https://arxiv.org/abs/1602.01783 |
| Trust Region Policy Optimization (TRPO) | Monotonic improvement via trust regions; prevents destructive updates in policy optimization. | Trust Region Policy Optimization | John Schulman et al. | 2015 | https://arxiv.org/abs/1502.05477 |
| Proximal Policy Optimization (PPO) | Simpler surrogate objective with clipping; widely used for its robustness and ease of implementation. | Proximal Policy Optimization Algorithms | John Schulman et al. | 2017 | https://arxiv.org/abs/1707.06347 |
| Actor-Critic with Experience Replay (ACER) | Off-policy actor-critic with replay; improves sample efficiency in policy-based methods. | Sample Efficient Actor-Critic with Experience Replay | Ziyu Wang et al. | 2016 | https://arxiv.org/abs/1611.01224 |
| Soft Actor-Critic (SAC) | Entropy-regularized off-policy; promotes exploration and handles continuous actions effectively. | Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor | Tuomas Haarnoja et al. | 2018 | https://arxiv.org/abs/1801.01290 |

#### 4. **Deterministic and Off-Policy Actor-Critic Methods**
These are key for continuous control, addressing exploration challenges in actor-critic frameworks.

| Method | Why Cornerstone? | Seminal Paper Title | Authors | Year | Link |
|--------|------------------|---------------------|---------|------|------|
| Deterministic Policy Gradient (DPG) | Deterministic policies for off-policy learning; foundation for continuous action spaces. | Deterministic Policy Gradient Algorithms | David Silver et al. | 2014 | http://proceedings.mlr.press/v32/silver14.pdf |
| Deep Deterministic Policy Gradient (DDPG) | Deep extension of DPG; uses actor-critic with replay for robotics and control tasks. | Continuous Control with Deep Reinforcement Learning | Timothy P. Lillicrap et al. | 2015 | https://arxiv.org/abs/1509.02971 |
| Twin Delayed DDPG (TD3) | Reduces overestimation with twin critics and delayed updates; improves DDPG stability. | Addressing Function Approximation Error in Actor-Critic Methods | Scott Fujimoto et al. | 2018 | https://arxiv.org/abs/1802.09477 |

#### 5. **Model-Based and Advanced Extensions**
These incorporate environment models for planning, essential for data efficiency and transfer.

| Method | Why Cornerstone? | Seminal Paper Title | Authors | Year | Link |
|--------|------------------|---------------------|---------|------|------|
| Model-Based Policy Optimization (MBPO) | Uses learned models for rollouts; bridges model-based and model-free for sample-efficient RL. | When to Trust Your Model: Model-Based Policy Optimization | Michael Janner et al. | 2019 | https://arxiv.org/abs/1906.08253 |
| MuZero | Model-based planning without given rules; unifies tree search with learned models for games. | Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model | Julian Schrittwieser et al. | 2020 | https://arxiv.org/abs/1911.08265 |
| Options Framework (Hierarchical RL) | Temporal abstractions for hierarchy; enables sub-policies for complex, long-horizon tasks. | Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning | Richard S. Sutton et al. | 1999 | https://www.sciencedirect.com/science/article/abs/pii/S0004370299000521 |

#### 6. **Hybrid RL Methods**
These combine RL with other techniques (e.g., offline/online data, imitation, meta-learning, or LLMs) to improve efficiency, generalization, and alignment. They are cornerstones for modern applications like autonomous systems and LLM fine-tuning.

| Method | Why Cornerstone? | Seminal Paper Title | Authors | Year | Link |
|--------|------------------|---------------------|---------|------|------|
| Hybrid Offline-Online RL | Combines offline datasets with online exploration; addresses sample efficiency in real-world RL without full resets. | Hybrid RL: Using Both Offline And Online Data Can Make RL Efficient | Gen Li et al. | 2023 | https://arxiv.org/abs/2210.06718 |
| Hybrid Actor-Critic in Parameterized Action Space | Merges discrete and continuous actions in actor-critic; foundational for complex action spaces in robotics and games. | Hybrid Actor-Critic Reinforcement Learning in Parameterized Action Space | Yichen Lu et al. | 2019 | https://arxiv.org/abs/1903.01344 |
| Generative Adversarial Imitation Learning (GAIL) | Hybrids RL with imitation via adversarial training; enables learning from demonstrations without explicit rewards. | Generative Adversarial Imitation Learning | Jonathan Ho and Stefano Ermon | 2016 | https://arxiv.org/abs/1606.03476 |
| Model-Agnostic Meta-Learning (MAML) for RL | Combines meta-learning with RL for fast adaptation; key for few-shot RL in varying tasks. | Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks | Chelsea Finn et al. | 2017 | https://arxiv.org/abs/1703.03400 |
| Reinforcement Learning from Human Feedback (RLHF) | Hybrid RL with human/AI preferences for LLM alignment; foundational for safe, value-aligned agents. | Fine-Tuning Language Models from Human Preferences | Daniel M. Ziegler et al. | 2019 | https://arxiv.org/abs/1909.08593 |
| Hybrid RL with LLMs (e.g., for Reasoning) | Integrates RL, imitation, and meta-learning for LLM reasoners; enables dynamic balancing of exploration and imitation. | Hybrid Learning Paradigms for LLM Reasoners: Combining Reinforcement, Imitation, and Meta-Learning | Micheal Lee | 2025 | https://www.researchgate.net/publication/394515116_Hybrid_Learning_Paradigms_for_LLM_Reasoners_Combining_Reinforcement_Imitation_and_Meta-Learning_Lee_Micheal |

#### 7. **Offline RL Methods** 
These learn from fixed datasets, vital for applied settings like healthcare or finance where real-time trials are risky.

| Method | Why Cornerstone? | Seminal Paper Title | Authors | Year | Link |
|--------|------------------|---------------------|---------|------|------|
| Conservative Q-Learning (CQL) | Prevents overestimation in offline data; enables safe policy learning from suboptimal datasets. | Conservative Q-Learning for Offline Reinforcement Learning | Aviral Kumar et al. | 2020 | https://arxiv.org/abs/2006.04779 |
| Batch-Constrained Q-Learning (BCQ) | Constrains actions to dataset distribution; foundational for offline value-based RL in robotics. | Off-Policy Deep Reinforcement Learning without Exploration | Scott Fujimoto et al. | 2018 | https://arxiv.org/abs/1812.02900 |

#### 8. **Multi-Agent RL (MARL) Methods** 
These handle interactions among multiple agents, key for practical applications like traffic simulation or team robotics.

| Method | Why Cornerstone? | Seminal Paper Title | Authors | Year | Link |
|--------|------------------|---------------------|---------|------|------|
| QMIX | Monotonic value decomposition for cooperative MARL; enables centralized training with decentralized execution. | QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning | Tabish Rashid et al. | 2018 | https://arxiv.org/abs/1803.11485 |
| Multi-Agent DDPG (MADDPG) | Extends DDPG to multi-agent; handles competitive/cooperative settings with centralized critics. | Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments | Ryan Lowe et al. | 2017 | https://arxiv.org/abs/1706.02275 |

#### 9. **Exploration-Focused Methods** 
These add intrinsic motivations to improve discovery in sparse-reward environments, commonly used in exploration-heavy tasks like navigation.

| Method | Why Cornerstone? | Seminal Paper Title | Authors | Year | Link |
|--------|------------------|---------------------|---------|------|------|
| Intrinsic Curiosity Module (ICM) | Generates intrinsic rewards based on prediction error; encourages curiosity-driven exploration. | Curiosity-driven Exploration by Self-supervised Prediction | Deepak Pathak et al. | 2017 | https://arxiv.org/abs/1705.05363 |
| Random Network Distillation (RND) | Uses novelty from random networks as intrinsic reward; simple and effective for hard-exploration problems. | Exploration by Random Network Distillation | Yuri Burda et al. | 2018 | https://arxiv.org/abs/1810.12894 |

#### 10. **Distributed and Scaling Methods** 
These enable large-scale, parallel training, essential for applied research in industry-scale AI.

| Method | Why Cornerstone? | Seminal Paper Title | Authors | Year | Link |
|--------|------------------|---------------------|---------|------|------|
| IMPALA | Importance-weighted actor-learner architecture; scales RL with asynchronous, distributed actors. | IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures | Lasse Espeholt et al. | 2018 | https://arxiv.org/abs/1802.01561 |
| APEX | Asynchronous prioritized experience replay; combines DQN with distributed workers for efficiency. | Distributed Prioritized Experience Replay | Dan Horgan et al. | 2018 | https://arxiv.org/abs/1803.00933 |

### Advice for Becoming an Applied AI Researcher
To transition from mastering these methods to applied research:
- **Implement and Experiment**: Code these in practical environments (e.g., Gymnasium for games, MuJoCo for robotics). Apply them to real datasets from Kaggle or Roboflow for domains like autonomous driving or recommendation systems.
- **Focus on Applications**: Study use cases in robotics (e.g., manipulation with SAC), finance (portfolio optimization with offline RL), or healthcare (treatment planning with MARL). Contribute to open-source like RLlib or Hugging Face's RL tools.
- **Build Skills Beyond RL**: Learn deployment (e.g., TensorFlow Serving), ethics/safety, and integration with other AI (e.g., RL + CV for vision-based agents). Read surveys on RL applications and join communities like Reddit's r/reinforcementlearning.
