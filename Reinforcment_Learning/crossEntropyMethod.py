import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

import argparse
from collections import namedtuple

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--batch_size", default=32, type=int)
arg_parser.add_argument("--percentile", default=70, type=int)

args = arg_parser.parse_args()
BATCH_SIZE = args.batch_size
PERCENTILE = args.percentile

model = keras.Sequential([
    keras.layers.Dense(6),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(3, activation="softmax")
])


Episode = namedtuple("Episode", ["steps", "total_reward"])
EpisodeStep = namedtuple("EpisodeStep", ["observation", "action"])

env = gym.make("Acrobot-v1", render_mode="human")

def generate_batch(env, model, batch_size):
    batch = []
    steps = []
    episode_reward = 0
    obs, _ = env.reset()
    while True:
            action_probs = model(obs[np.newaxis]).numpy()[0]
            action = np.random.choice([0, 1, 2], p=action_probs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            step = EpisodeStep(obs, action)
            steps.append(step)
            obs = next_obs
            if terminated or truncated:
                batch.append(Episode(steps, episode_reward))
                episode_reward = 0
                steps = []
                next_obs, _ = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
            obs = next_obs

def filter_episode(batch, percentile):
    rewards = list(map(lambda x: x.total_reward, batch))
    rewards_mean = np.mean(rewards)
    rewards_bounds = np.percentile(rewards, percentile)

    train_obs = []
    train_act = []
    for steps, reward in batch:
        if reward > rewards_bounds:
            train_obs.extend(map(lambda x: x.observation, steps))
            train_act.extend(map(lambda x: x.action, steps))
    return train_obs, train_act, rewards_mean, rewards_bounds

optimizer = keras.optimizers.Adam(learning_rate=0.005)
loss_fn = keras.losses.CategoricalCrossentropy()

@tf.function
def compute_apply_gradient(model, obs_v, y_target):
    with tf.GradientTape() as tape:
        y_pred = model(obs_v)
        loss = loss_fn(y_target, y_pred)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
try:   
    for i, episodes in enumerate(generate_batch(env, model, BATCH_SIZE)):
        obs_v, act_v, rewards_mean, rewards_boundary = filter_episode(episodes, PERCENTILE)
        if len(obs_v) == 0:
            continue
        y_target = keras.ops.one_hot(act_v, 3)
        obs_v = tf.reshape(tf.Variable(obs_v), (-1, 6))
        compute_apply_gradient(model, obs_v, y_target)
        print(f"Iteration {i} _ Rewards_Mean {rewards_mean} _ Rewards Boundary {rewards_boundary}")
except KeyboardInterrupt as e:
    print("Stoping experiment")
else:
    env.close()