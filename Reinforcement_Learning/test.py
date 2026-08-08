from a2c import Actor2Critic, Agent
import gymnasium as gym
from tensorflow import keras
import ale_py

gym.register_envs(ale_py)
env = gym.make("ALE/BattleZone-v5")
test_env = gym.make("ALE/BattleZone-v5", render_mode="rgb_array")
model = Actor2Critic(env.observation_space.shape, 18)
gamma = 0.99
critic_loss_fn = keras.losses.Huber()
optimizer = keras.optimizers.Nadam(global_clipnorm=0.5)
agent = Agent(env, test_env, model, optimizer, critic_loss_fn, gamma)

import timeit
import psutil
import os
import numpy  as np
import tensorflow as tf
process = psutil.Process(os.getpid())

