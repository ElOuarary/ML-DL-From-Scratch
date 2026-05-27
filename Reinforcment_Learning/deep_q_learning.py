from collections import deque
import gymnasium as gym
import numpy as np
from tensorflow import keras

replay_buffer = deque(maxlen=2000)

env = gym.make("Acrobot-v1", render="human")

input_shape = env.observation_space.shape
n_outputs = env.action_space.shape

model = keras.Sequential([
    keras.layers.Dense(32, activation="elu", input_shape=input_shape),
    keras.layers.Dense(32, activation="elu"),
    keras.layers.Dense(n_outputs)
])

def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)
    else:
        Q_values = model.predict(state[np.newaxis])[0]
        return Q_values.argmax()
    
def sample_experiences(batch_size):
    indices = np.random.randint(len(batch_size), batch_size)
    batch = [replay_buffer[index] for index in indices]
    return [
        np.array([experience[field_index] for experience in batch]) for field_index in range(6)
    ] # [states, actions, rewards, next_states, terminateds, truncateds]
    
def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, terminated, truncaated, info = env.step(action)
    replay_buffer.append((state, action, next_state, reward, terminated, truncaated))
    return next_state, reward, terminated, truncaated, info