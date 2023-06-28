import gym
from gym import spaces
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class StopTrainingOnDoneCallback(BaseCallback):
    def __init__(self, env_done_threshold, verbose=0):
        super(StopTrainingOnDoneCallback, self).__init__(verbose)
        self.env_done_threshold = env_done_threshold

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.env_done_threshold:  # Stop training after a certain number of timesteps
            return False
        if self.locals.get("done", False):  # Check if the environment is done
            return False
        return True


class TradingEnv(gym.Env):
    def __init__(self, data, window: int):
        super(TradingEnv, self).__init__()
        self.window = window
        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # buy, sell, do nothing
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window, 5))  # OHLCV data

        # Initialize state
        self.data = data
        self.time_step = 0
        self.holding = False
        self.cost_basis = 0
        self.cumulative_reward = 0
        self.buy_action_count = 0
        self.sell_action_count = 0
        self.no_action_count = 0

    def step(self, action):
        # Execute one time step within the environment
        # Use close price
        current_price = float(self.data.iloc[self.time_step, 3])

        if action == 0:  # buy
            if not self.holding:
                self.holding = True
                self.cost_basis = current_price
            reward = 0
            self.buy_action_count += 1
        elif action == 1:  # sell
            if self.holding:
                self.holding = False
                reward = current_price - self.cost_basis
                self.cumulative_reward += reward
            else:
                reward = 0
            self.sell_action_count += 1
        else:  # do nothing
            reward = 0
            self.no_action_count += 1

        self.time_step += 1

        done = self.time_step >= len(self.data) - self.window
        obs = self.data[self.time_step:min(self.time_step+self.window, len(self.data))]

        if done:
            print(f'Environment Done: cumulative_reward: {self.cumulative_reward}, '
                  f'buy: {self.buy_action_count}, '
                  f'sell: {self.sell_action_count}, '
                  f'no: {self.no_action_count}')
            self.reset()

        return obs, reward, done, {}

    def reset(self, **kwargs):
        # Reset the state of the environment to an initial state
        self.time_step = 0
        self.cumulative_reward = 0
        self.buy_action_count = 0
        self.sell_action_count = 0
        self.no_action_count = 0
        self.holding = False
        self.cost_basis = 0
        return self.data[self.time_step:min(self.time_step+10, len(self.data))]
