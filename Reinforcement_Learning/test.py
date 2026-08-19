import os
os.environ["MUJOCO_GL"] = "egl"
import gymnasium as gym
env = gym.make("HumanoidStandup-v5", render_mode="rgb_array")

import imageio

import numpy as np
from tensorflow.keras.models import load_model
from a2c_gaussian import A2C_Guassian


model = A2C_Guassian(env.observation_space.shape, env.action_space.shape[0])

model.build(env.observation_space.sample()[np.newaxis])

obs, _ = env.reset()
frames = []
total_reward = 0
while True:
    frame = env.render()
    frames.append(frame)
    optimal_action, _, _ = model(obs[np.newaxis])
    states, reward, _, truncated, _ = env.step(optimal_action[0].numpy())
    total_reward += reward

    if truncated:
        break

imageio.mimsave("HumanoidStandup/demo-2.gif", frames, fps=30)
