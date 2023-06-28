from stable_baselines3 import PPO, A2C
from trading.moving_average.data.bitcoin import get_bitcoin_data
from trading.moving_average.env import TradingEnv, StopTrainingOnDoneCallback

data = get_bitcoin_data()
data = data.drop('Open time', axis=1)

model = PPO('MlpPolicy', TradingEnv(data, window=10), verbose=0)

callback = StopTrainingOnDoneCallback(env_done_threshold=100000)  # Adjust the threshold as needed
model.learn(total_timesteps=1000000, progress_bar=True)

# Evaluate the agent
# mean_reward, std_reward = evaluate_policy(model, TradingEnv(data, window=10), n_eval_episodes=10)

# print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
