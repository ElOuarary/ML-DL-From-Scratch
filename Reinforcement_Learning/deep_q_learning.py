import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

import argparse
from datetime import datetime
from collections import deque, namedtuple

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.001)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--gamma", type=float, default=0.95)
parser.add_argument("--buffer-size", type=int, default=2_000)

STEP = namedtuple("Step", field_names=("state", "action", "reward", "next_state", "done"))

class Agent:
    def __init__(self, env, epsilon, gamma, net, tg_net, loss_fn, optimizer, buffer_size):
        self.env = env
        self.state, _ = self.env.reset()
        self.action_space = env.action_space.n
        self.epsilon = epsilon
        self.gamma = gamma
        self.net = net
        self.tg_net = tg_net
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.replay_buffer = deque(maxlen=buffer_size)
        self.low  = tf.constant([-2.5, -2.5, -10., -10., -6.2831855, -10., -0., -0.])
        self.high = tf.constant([ 2.5,  2.5,  10.,  10.,  6.2831855,  10.,  1.,  1.])

    def explore(self):
        if np.random.sample() < max(0.01, self.epsilon):
            action = self.env.action_space.sample()
        else:
            action = tf.argmax(self.net(self.state[np.newaxis]), axis=-1)
            action = action.numpy()[0]
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        self.replay_buffer.append(STEP(state=self.state, action=action, reward=reward, next_state=next_state, done=0 if terminated or truncated else 1))
        if terminated or truncated:
            self.state, _ = self.env.reset()
        else:
            self.state = next_state
    
    def sample_batch(self, batch_size):
        indices = np.random.choice(range(len(self.replay_buffer)), batch_size, replace=False)
        states, actions, rewards, next_states, done = zip(*[self.replay_buffer[idx] for idx in indices])
        states, actions, rewards, next_states, done = tf.constant(states), tf.constant(actions), tf.constant(rewards, dtype=tf.float32), tf.constant(next_states, dtype=tf.float32), tf.constant(done, dtype=tf.float32)
        states, rewards, next_states, done = tf.reshape(states, (batch_size, -1)), tf.reshape(rewards, (batch_size, -1)), tf.reshape(next_states, (batch_size, -1)), tf.reshape(done, (batch_size, -1))
        return states, actions, rewards, next_states, done

    def normalize_state(self, state):
        return 2 * (state - self.low) / (self.high - self.low) - 1

    @tf.function
    def compute_loss(self, state, reward, action, next_state, done):
        next_state = self.normalize_state(next_state)
        state = self.normalize_state(state)
        next_state_value = tf.reduce_max(self.tg_net(next_state), axis=-1)
        
        q_value_target = reward + self.gamma * done * next_state_value
        mask = tf.one_hot(action, self.action_space)
        with tf.GradientTape() as tape:
            q_value = self.net(state)
            q_value_masked = tf.reduce_sum(mask * q_value, axis=-1, keepdims=True)
            loss = self.loss_fn(q_value_target, q_value_masked)
        gradients = tape.gradient(loss, self.net.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,self.net.trainable_variables))
        return loss
    
    def train_model(self, batch_size):
        states, actions, rewards, next_states, done = self.sample_batch(batch_size)
        loss = self.compute_loss(states, rewards, actions, next_states, done)
        return loss

    def test(self, test_env):
        state, _ = test_env.reset()
        total_reward = 0
        while True:
            state = self.normalize_state(state)
            q_values = self.net(state[np.newaxis])
            action = tf.argmax(q_values, axis=-1).numpy()[0]
            next_state, reward, terminated, truncated, _ = test_env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
            state = next_state
        return total_reward

def main():
    args = parser.parse_args()
    alpha = args.alpha
    batch_size = args.batch_size
    gamma = args.gamma
    buffer_size = args.buffer_size

    env = gym.make("LunarLander-v3")
    test_env = gym.make("LunarLander-v3")
    model = keras.Sequential([
        keras.layers.InputLayer(env.observation_space.shape),
        keras.layers.Dense(256),
        keras.layers.Dense(4)
    ])
    tg_model = keras.models.clone_model(model)
    tg_model.set_weights(model.get_weights())

    loss_fn = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Nadam(learning_rate=alpha)
    agent = Agent(env, 1, gamma, model, tg_model, loss_fn, optimizer, buffer_size)

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_logs_dir = "logs/dqn/LunarLander/train/" + current_time
    test_logs_dir = "logs/dqn/LunarLander/test/" + current_time

    train_summary_writer = tf.summary.create_file_writer(train_logs_dir)
    test_summary_writer = tf.summary.create_file_writer(test_logs_dir)
    
    try:
        for i in range(1, 10_001):
            agent.explore()
            agent.epsilon = max(1 - i / 9_000, 0.01)

            if len(agent.replay_buffer) >= 50:
                loss = agent.train_model(batch_size)

                total_rewards = 0
                for _ in range(10):
                    total_rewards += agent.test(test_env)

                mean_reward = total_rewards / 10
                with train_summary_writer.as_default():
                    tf.summary.scalar("train_loss", loss, step=i)

                with test_summary_writer.as_default():
                    tf.summary.scalar("test_mean_reward", mean_reward, step=i)

                if mean_reward > 200:
                    print("Problem Solved")
                    break
                

                print(f"Iteration: {i} - Loss {loss} - Mean Reward: {mean_reward}")

            if i % 100 == 0:
                print("Target Model Updating")
                agent.tg_net.set_weights(agent.net.get_weights())

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        test_env.close()

if __name__ == "__main__":
    main()