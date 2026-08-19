# import os
# os.environ["MUJOCO_GL"] = "egl"
# import gymnasium as gym
# env = gym.make("HumanoidStandup-v5", render_mode="rgb_array")

# import imageio

# import numpy as np
# from tensorflow.keras.models import load_model
# from a2c_gaussian import A2C_Guassian


# model = A2C_Guassian(env.observation_space.shape, env.action_space.shape[0])

# model.build(env.observation_space.sample()[np.newaxis])

# obs, _ = env.reset()
# frames = []
# total_reward = 0
# while True:
#     frame = env.render()
#     frames.append(frame)
#     optimal_action, _, _ = model(obs[np.newaxis])
#     states, reward, _, truncated, _ = env.step(optimal_action[0].numpy())
#     total_reward += reward

#     if truncated:
#         break

# imageio.mimsave("HumanoidStandup/demo-2.gif", frames, fps=30)

import tensorflow as tf

mu = tf.constant([[1.0, 2.0], [1.0, 2.0]], dtype=tf.float32)
st = tf.constant([[1.0, 2.0], [1.0, 2.0]], dtype=tf.float32)

print(tf.clip_by_value(tf.random.normal((2,), mean=mu, stddev=st), clip_value_min=0, clip_value_max=1))