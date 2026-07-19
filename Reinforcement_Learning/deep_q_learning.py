from collections import deque
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow as tf

replay_buffer = deque(maxlen=2000)

env = gym.make("Acrobot-v1", render_mode="human")

input_shape = env.observation_space.shape
n_outputs = 3

model = keras.Sequential([
    keras.layers.Dense(32, activation="elu", input_shape=input_shape),
    keras.layers.Dense(32, activation="elu"),
    keras.layers.Dense(n_outputs)
])

target = keras.models.clone_model(model)
target.set_weights(model.get_weights())

def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)
    else:
        Q_values = model.predict(state[np.newaxis])[0]
        return Q_values.argmax()
    
def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in indices]
    return [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(6)
    ] # [states, actions, rewards, next_states, dones, truncateds]
    
def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, truncated, terminated, info = env.step(action)
    replay_buffer.append([state, action, reward, next_state, terminated, truncated])
    return next_state, reward, truncated, terminated, info

batch_size = 32
discount_factor = .95
optimizer = keras.optimizers.Nadam(learning_rate=1e-2)
loss_fn = keras.losses.mse

def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, terminateds, truncateds = experiences
    next_Q_values = model.predict(next_states, verbose=False)
    best_next_actions = next_Q_values.argmax(axis=1)
    next_mask = tf.one_hot(best_next_actions, n_outputs).numpy()
    max_next_Q_values = (target.predict(next_states, verbose=0) * next_mask).sum(axis=1)
    # max_next_Q_values = next_Q_values.max(axis=1)
    runs = 1 - (terminateds | truncateds)
    target_Q_values = rewards + runs * discount_factor * max_next_Q_values
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, n_outputs)
    
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
for episode in range(600):
    sum_rewards = []
    obs, info = env.reset()
    all_rewards = 0
    for step in range(500):
        print(f"Episode: {episode} _ Step: {step}")
        epsilon = max(1 - episode / 500, .01)
        obs, reward, terminated, truncated, info = play_one_step(env, obs, epsilon)
        all_rewards += reward  
        if terminated or truncated:
            break
        
    sum_rewards.append(all_rewards)
    all_rewards = 0
    
    if episode % 50 == 0:
        target.set_weights(model.get_weights())
        
    if episode > 50:
        training_step(batch_size)
        
plt.plot(range(1, 601), sum_rewards)
plt.show()

model.save("DQL_acrobat_policy.keras") 