import gymnasium as gym
import numpy as np
from tensorflow import keras
import tensorflow as tf

env = gym.make("Acrobot-v1", render_mode="human")
obs, info = env.reset()

episode_over = False
totals = 0

model = keras.Sequential([
    keras.layers.InputLayer(6),
    keras.layers.Dense(12, activation="relu"),
    keras.layers.Dense(3, activation="softmax")
])

def play_one_step(env, obs, model, loss_fn):
    with tf.GradientTape() as tape:
        y_pred = model(obs[np.newaxis])
        action = np.argmax(y_pred)
        y_target = tf.zeros(y_pred.shape)
        y_target[action] = 1
        loss = loss_fn(y_target, y_pred)
    
    grad = tape.gradient(loss, model.trainable_variables)
    obs, reward, terminated, truncated, info = env.step(action)
    return obs, reward, terminated, truncated, grad

def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
    all_rewards = []
    all_gradients = []
    for episode in range(n_episodes):
        obs, info = env.reset()
        rewards = []
        gradients = []
        for step in range(n_max_steps):
            obs, reward, terminated, truncated, grad = play_one_step(env, obs, model, loss_fn)
            rewards.append(reward)
            gradients.append(grad)
            if terminated or truncated:
                break      
        all_rewards.append(rewards)
        all_gradients.append(gradients)
    return all_rewards, all_gradients

def discount_rewards(rewards, discount_factor):
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        discounted[step] += discounted[step + 1] * discount_factor
    return discounted

def discount_and_normalize_rewards(all_rewards, dicount_factor):
    pass

n_iterations = 150
n_episode_per_iteration = 10
n_max_steps = 500
discount_factor = 0.95

optimizers = keras.optimizers.Nadam(learning_rate=0.01)
loss_fn = keras.losses.categorical_crossentropy

while not episode_over:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    totals += reward
    episode_over = terminated or truncated
    
print(totals)
env.close()