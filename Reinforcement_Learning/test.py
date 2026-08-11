from a2c import Actor2Critic, Agent
import gymnasium as gym
import ale_py

import tensorflow as tf
from tensorflow import keras

gym.register_envs(ale_py)
env = gym.make("ALE/BattleZone-v5")
test_env = gym.make("ALE/BattleZone-v5")

model = Actor2Critic(env.observation_space.shape, 18)
gamma = 0.99

critic_loss_fn = keras.losses.Huber()
optimizer = keras.optimizers.Nadam(learning_rate=0.0015, clipnorm=0.1)

agent = Agent(env, test_env, model, optimizer, critic_loss_fn, gamma, 100)

for _ in range(5):
    episode_reward, loss, actor_loss, critic_loss, iteration_time = agent.learn_from_episode()
    print(iteration_time)