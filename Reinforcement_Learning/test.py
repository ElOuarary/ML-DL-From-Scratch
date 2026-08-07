import gymnasium as gym
import ale_py
from a2c import Actor2Critic, full_train_step
import tensorflow as tf
import cProfile
import numpy as np

gym.register_envs(ale_py)

env = gym.make("ALE/BattleZone-v5")
model = Actor2Critic(env.observation_space.shape, 18)

obs, _ = env.reset()

for _ in range(10):
    print(model(obs[np.newaxis]))