import gymnasium as gym
import numpy as np
from tensorflow import keras
import tensorflow as tf

env = gym.make("Acrobot-v1", render_mode="human")
obs, info = env.reset()

def play_one_step(env, obs, model, loss_fn):
    with tf.GradientTape() as tape:
        y_pred = model(obs[np.newaxis])
        action = np.random.choice(len(y_pred[0]), p=y_pred[0].numpy())
        y_target = np.zeros(y_pred.shape)
        y_target[0, action] = 1
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
            print(f"Episode: {episode} _ Step: {step}")
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

def discount_and_normalize_rewards(all_rewards, discount_factor):
    all_discounted_rewards = [discount_rewards(reward, discount_factor) for reward in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    mean = flat_rewards.mean()
    std = flat_rewards.std()
    return [(discounted_reward - mean) / std for discounted_reward in all_discounted_rewards]
    
n_iterations = 150
n_episode_per_iteration = 10
n_max_steps = 500
discount_factor = 0.95


model = keras.Sequential([
    keras.layers.Dense(6, activation="relu"),
    keras.layers.Dense(3, activation="softmax")
])

optimizer = keras.optimizers.Nadam(learning_rate=0.01)
loss_fn = keras.losses.categorical_crossentropy

for iteration in range(n_iterations):
    print(f"Iteration: {iteration}")
    all_rewards, all_gradients = play_multiple_episodes(env, n_episode_per_iteration, n_max_steps, model, loss_fn)
    all_discounted_normalized_rewards = discount_and_normalize_rewards(all_rewards, discount_factor)
    
    all_mean_grads = []
    for var_index in range(len(model.trainable_variables)):
        mean_grad = tf.reduce_mean(
            [
                final_reward * all_gradients[episode_index][step][var_index]
                for episode_index, final_rewards in enumerate(all_discounted_normalized_rewards)
                    for step, final_reward in enumerate(final_rewards)
            ]
        )
        all_mean_grads.append(mean_grad)
    optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))
    
env.close()