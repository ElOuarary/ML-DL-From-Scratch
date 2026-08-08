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

from scipy.signal import lfilter

@tf.function
def compute_returns_5(episode_rewards) -> tf.Tensor:
    n = tf.shape(episode_rewards)[0]
    discount = 0.99 ** tf.cast(tf.range(n), tf.float32)
    returns = tf.cumsum(tf.reverse(episode_rewards * discount, axis=[0]))
    returns = tf.reverse(returns, axis=[0]) / discount
    mean = tf.reduce_mean(returns)
    std = tf.math.reduce_std(returns)
    return (returns - mean) / (std + 1e-8)


rewards = np.random.randn(1000).tolist()

rewards = np.asarray(rewards, dtype=np.float32)
print(timeit.timeit(lambda: agent.compute_returns(rewards), number=100, globals=globals()))

tf_rewards = tf.random.normal((1000,), dtype=tf.float32)
print(timeit.timeit(lambda:compute_returns_5(tf_rewards), number=100, globals=globals()))


# start = process.memory_info().rss / 1e6
# print(timeit.timeit("agent.learn_from_episode()", number=10, globals=globals()))
# end = process.memory_info().rss / 1e6

# print(f"RSS: {end} - Delta: {end-start}")