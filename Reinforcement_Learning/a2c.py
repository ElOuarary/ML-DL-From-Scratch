import gymnasium as gym
import ale_py
import numpy as np
import tensorflow as tf
from tensorflow import keras

gym.register_envs(ale_py)

class Actor2Critic(keras.Model):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.shared_network = keras.models.Sequential([
            keras.layers.Conv2D(32, kernel_size=(3, 3),activation="relu", input_shape=observation_space),
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dense(256, activation="relu")
        ])
        self.actor = keras.layers.Dense(18, activation="softmax")
        self.critic = keras.layers.Dense(1)

    @tf.function
    def call(self, obs):
        x = self.shared_network(obs)
        return self.actor(x), self.critic(x)

@tf.numpy_function(Tout=[tf.float32, tf.int32, tf.int32])
def env_step(action):
    next_obs, reward, terminated, truncated, _ = env.step(action)
    return (
        next_obs.astype(np.float32),
        np.array(reward, np.int32),
        np.array(terminated | truncated, np.int32),
    )

def run_episode(initial_state, model):
    action_probas = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
    initial_state_shape = initial_state.shape
    state = initial_state
    for _ in tf.range(100_000):
        state = tf.expand_dims(state, axis=0)
        action_logits, value = model(state)
        action = tf.random.categorical(action_logits, 1)[0, 0]
        action_proba = tf.nn.softmax(action_logits)

        values = values.write(values.size(), value)
        action_probas = action_probas.write(action_probas.size(), action_proba)
        state, reward, done = env_step(action)
        rewards = rewards.write(rewards.size(), reward)
        state.set_shape(initial_state_shape)
        if tf.cast(done, tf.bool):
            break

    action_probas = action_probas.stack()
    values = values.stack()
    rewards = rewards.stack()
    return action_probas, values, rewards

def get_expected_return(rewards, gamma, standardize=True):
    returns = tf.TensorArray(dtype=tf.float32, size=tf.shape(rewards)[0])
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(tf.shape(rewards)[0]):
        returns = returns.write(i, rewards[i] + gamma * discounted_sum)
        discounted_sum = returns.read(i)
        discounted_sum.set_shape(discounted_sum_shape)
    returns = returns.stack()[::-1]
    if standardize:
        returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + 1e-8))
    return returns

huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

def compute_loss(action_probas, values, returns):
    advantage = returns - values
    action_log_probas = tf.math.log(action_probas)
    action_loss = -tf.math.reduce_sum(action_log_probas * advantage)
    value_loss = huber_loss(values, returns)
    return action_loss + value_loss

optimizer = keras.optimizers.Nadam()

@tf.function
def train_step(initial_step, model, optimizer, gamma):
    with tf.GradientTape() as tape:
        action_probas, values, rewards = run_episode(initial_step, model)
        returns = get_expected_return(rewards, gamma)
        action_probas, values, returns = [
            tf.expand_dims(x, axis=1) for x in [action_probas, values, returns]
        ]
        loss = compute_loss(action_probas, values, returns)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return tf.math.reduce_sum(rewards)

reward_treshold = 3000
max_episode = 100_000
gamma = 0.99
running_reward = 0
env = gym.make("ALE/BattleZone-v5")
model = Actor2Critic(env.observation_space.shape, 18)

for i in range(1, max_episode+1):
    initial_state, _ = env.reset()
    initial_state = tf.constant(initial_state, dtype=tf.float32)
    episode_reward = int(train_step(initial_state, model, optimizer, gamma))
    
    if i % 1000:
        print(f"Iteration {i} - Reward: {episode_reward}")

    if episode_reward > rewar_treshold:
        print(f"Problem Solved! Reward {episode_reward}")
        break
