import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

from collections import deque, namedtuple

ALPHA = 0.001
BATCH_SIZE = 32
GAMMA = 0.95
REPLAY_BUFFER_SIZE = 10_000


STEP = namedtuple("Step", field_names=("state", "action", "reward", "next_state", "done"))

class Agent:
    def __init__(self, env, epsilon, gamma, net, tg_net, loss_fn, optimizer):
        self.env = env
        self.state, _ = self.env.reset()
        self.action_space = env.action_space.n
        self.epsilon = epsilon
        self.gamma = gamma
        self.net = net
        self.tg_net = tg_net
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.replay_buffer = deque(maxlen=1000)

    def explore(self):
        if np.random.sample() < max(0.01, self.epsilon):
            action = self.env.action_space.sample()
        else:
            action = self.net(self.state[np.newaxis]).argmax()
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        self.replay_buffer.append(STEP(state=self.state, action=action, reward=reward, next_state=next_state, done=0 if terminated or truncated else 1))
        if terminated or truncated:
            self.state, _ = self.env.reset()
    
    def sample_batch(self, batch_size):
        indices = np.random.choice(range(len(self.replay_buffer)), batch_size, replace=False)
        states, actions, rewards, next_states, done = zip(*[self.replay_buffer[idx] for idx in indices])
        states, actions, rewards, next_states, done = tf.constant(states), tf.constant(actions), tf.constant(rewards, dtype=tf.float32), tf.constant(next_states, dtype=tf.float32), tf.constant(done, dtype=tf.float32)
        states, actions, rewards, next_states, done = tf.reshape(states, (batch_size, -1)), tf.reshape(actions, (batch_size, -1)), tf.reshape(rewards, (batch_size, -1)), tf.reshape(next_states, (batch_size, -1)), tf.reshape(done, (batch_size, -1))
        return states, actions, rewards, next_states, done
    
    def compute_loss(self, state, reward, action, next_state, done):
        next_state_value = tf.reduce_max(self.tg_net(next_state), axis=1)
        next_state_value = tf.reshape(next_state_value, (next_state_value.shape[0], -1))

        q_value_target = reward + self.gamma * done * next_state_value
        mask = tf.one_hot(action, self.action_space)
        with tf.GradientTape() as tape:
            q_value = self.net(state)
            q_value = tf.reduce_sum(q_value * mask, axis=1, keepdims=True)
            loss = tf.reduce_sum(self.loss_fn(q_value_target, q_value))
        gradients = tape.gradient(loss, self.net.trainable_variables)
        self.optimizer.apply(gradients,self.net.trainable_variables )
        return loss
    
    def train_model(self, batch_size):
        states, actions, rewards, next_states, done = self.sample_batch(batch_size)
        loss = self.compute_loss(states, rewards, actions, next_states, done)
        return loss

    def test(self, test_env):
        state, _ = test_env.reset()
        total_reward = 0
        while True:
            q_values = self.net(state[np.newaxis])
            action = tf.argmax(q_values).numpy()[0]
            next_state, reward, terminated, truncated, _ = test_env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
            state = next_state
        return total_reward

def main():
    env = gym.make("<")
    test_env = gym.make("LunarLander-v3")
    model = keras.Sequential([
        keras.layers.InputLayer(env.observation_space.shape),
        keras.layers.Dense(256),
        keras.layers. Dense(4)
    ])
    tg_model = tf.keras.models.clone_model(model)
    tg_model.set_weights(model.get_weights())

    loss_fn = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Nadam()
    agent = Agent(env, 1, 0.95, model, tg_model, loss_fn, optimizer)

    for i in range(10_000):
        agent.explore()

        if len(agent.replay_buffer) >= 50:
            loss = agent.train_model(32)

            total_rewards = 0
            for _ in range(20):
                total_rewards += agent.test(test_env)

            mean_reward = total_rewards / 20
            if mean_reward > 200:
                print("Problem Solved")
        
            print(f"Iteration: {i} - Loss {loss} - Mean Reward: {mean_reward}")

        if i % 50 == 0:
            print("Target Model Updating")
            agent.tg_net.set_weights(agent.net.get_weights())


if __name__ == "__main__":
    main()