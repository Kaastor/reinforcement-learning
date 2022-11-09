import numpy as np
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('Agg')

epochs = 2000
episode_len = 1000
number_of_bandits = 2
actions = 10
true_values = {}


# encapsulation for single bandit problem
class Bandit:
    def __init__(self, k_arm=10, epsilon=0.):
        self.arms = k_arm  # number of actions
        self.epsilon = epsilon

    def reset(self):
        pass

    def act(self):
        pass

    def step(self, action):
        pass


for i in range(0, number_of_bandits):
    true_values[i] = {}
    for j in range(1, actions):
        true_values[i][j] = np.random.normal()  # generate true value for action


def get_actual_reward(bandit, action):
    val = true_values[bandit][action]
    return np.random.normal(val, 1)


# TODO
def simulate(runs, time, bandits):
    pass


def figure_2_1():
    plt.violinplot(dataset=np.random.randn(200, 10) + np.random.randn(10))
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")
    plt.savefig('./images/figure_2_1.png')
    plt.close()


def figure_2_2(runs=2000, time=1000):
    epsilons = [0, 0.1, 0.01]
    bandits = [Bandit(epsilon=eps, sample_averages=True) for eps in epsilons]
    best_action_counts, rewards = simulate(runs, time, bandits)

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    for eps, rewards in zip(epsilons, rewards):
        plt.plot(rewards, label='$\epsilon = %.02f$' % (eps))
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    for eps, counts in zip(epsilons, best_action_counts):
        plt.plot(counts, label='$\epsilon = %.02f$' % (eps))
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.savefig('./images/figure_2_2.png')
    plt.close()


figure_2_1()
figure_2_2()

# TODO: zaimplementuj simple bandit algorytm dla 2000 runs 1000 epizod bez patrzenia na gotowca
#  narysuj 2.2
