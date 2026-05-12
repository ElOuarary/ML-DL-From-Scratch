import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()

print("First Obeservation: ", obs)

totals = []
for episode in range(500):
    episode_rewards = 0
    obs, info = env.reset(seed=episode)
    for step in range(200):
        obs, reward, done, truncated, info = env.step(1) if obs[2] > 0 else env.step(0)
        episode_rewards += reward
        
        if done or truncated:
            break
    totals.append(episode_rewards)
        
env.close()

print("Mean: ", np.mean(totals))
print("Standard Deviation: ", np.std(totals))
print("Min: ", np.min(totals))
print("Max: ", np.max())