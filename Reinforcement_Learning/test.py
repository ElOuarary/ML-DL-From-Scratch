import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import ale_py
import numpy as np
from utils import make_atari_env

gym.register_envs(ale_py)

env = SyncVectorEnv([make_atari_env("ALE/BattleZone-v5") for _ in range(8)])

env.reset()
obs, returns, terminated, truncated, _ = env.step(env.action_space.sample())

print(terminated | truncated)
print(np.all(terminated | truncated))