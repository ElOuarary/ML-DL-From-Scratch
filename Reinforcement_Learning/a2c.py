import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import ale_py
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tqdm

import imageio
from datetime import datetime
import gc
import psutil, os, csv
from collections import deque
from time import perf_counter
from typing import Tuple

from utils import make_atari_env

class MetricLog:
    def __init__(self, path="metrics.csv"):
        self.path = path
        self.process = psutil.Process(os.getpid())
        with open(self.path, "w") as f:
            writer = csv.writer(f)
            writer.writerow([
                "iteration", "step_time_s", "rss_mb", "cpu_percent"
            ])

    def log(self, iteration,step_time_s):
        rss = self.process.memory_info().rss / (1024 * 1024)
        cpu = self.process.cpu_percent()
        with open(self.path, "a") as f:
            writer = csv.writer(f)
            writer.writerow([
                iteration, step_time_s, rss, cpu
            ])

class Actor2Critic(keras.Model):
    def __init__(self, observation_space: int, action_space: int):
        super().__init__()
        self.shared_network = keras.models.Sequential([
            keras.layers.InputLayer(observation_space),
            keras.layers.Conv2D(32, kernel_size=8, strides=4,activation="relu"),
            keras.layers.Conv2D(64, kernel_size=4, strides=2, activation="relu"),
            keras.layers.Conv2D(64, kernel_size=3, strides=1, activation="relu"),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dense(256, activation="relu")
        ])
        self.actor = keras.layers.Dense(action_space)
        self.critic = keras.layers.Dense(1)

    def call(self, obs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.shared_network(obs)
        return self.actor(x), self.critic(x)

class Agent:
    def __init__(self, env, test_env, model, optimizer, critic_loss_fn, gamma, n_steps=5):
        self.env = env
        self.current_states, _ = env.reset()
        self.test_env = test_env
        self.model = model
        self.critic_loss_fn = critic_loss_fn
        self.optimizer = optimizer
        self.gamma = gamma
        self.n_steps = n_steps

        self.states = tf.Variable(tf.zeros_initializer()(shape=(self.n_steps, 8, 84, 84, 4), dtype=tf.float32))
        self.actions = tf.Variable(tf.zeros_initializer()(shape=(self.n_steps, self.env.num_envs), dtype=tf.int32))
        self.rewards = tf.Variable(tf.zeros_initializer()(shape=(self.n_steps, self.env.num_envs), dtype=tf.float32))
        self.dones = tf.Variable(tf.zeros_initializer()(shape=(self.n_steps, self.env.num_envs), dtype=tf.bool))
        self.boostrapped_values = tf.Variable(tf.zeros_initializer()(shape=(self.env.num_envs,), dtype=np.float32))
        self.train_returns = tf.Variable(tf.zeros_initializer()(shape=self.rewards.shape, dtype=np.float32))
                

    @tf.function
    def predict(self, state):
        action_logits, _ = self.model(state)
        action = tf.random.categorical(action_logits, num_samples=1, dtype=tf.int32)
        return action

    def _step_env(self, actions: np.ndarray):
        next_states, rewards, terminated, truncated, _ = self.env.step(actions)
        return next_states.astype(np.float32), rewards.astype(np.float32), np.logical_or(terminated, truncated)

    @tf.function
    def collect_rollout(self, states: tf.Tensor, train_states: tf.Tensor, train_actions: tf.Tensor, train_rewards: tf.Tensor, train_dones: tf.Tensor, boostrapped_values: tf.Tensor, train_returns: tf.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        for i in tf.range(self.n_steps):
            states_transpose = tf.transpose(states, perm=[0, 2, 3, 1]) / 255.0
            actions = self.predict(states_transpose)[:, 0]
            next_states, rewards, dones = tf.numpy_function(
                func=self._step_env,
                inp=[actions],
                Tout=[tf.float32, tf.float32, tf.bool]
            )
            train_states[i].assign(states_transpose)
            train_actions[i].assign(actions)
            train_rewards[i].assign(rewards)
            train_dones[i].assign(dones)

            states.assign(next_states)

        self.current_states = states

        next_states = tf.transpose(next_states, perm=(0, 2, 3, 1)) / 255.0
        _, next_state_value = self.model(next_states)
        boostrapped_values.assign(next_state_value[:, 0]) 

        for i in tf.reverse(tf.range(self.n_steps), axis=[0]):
            boostrapped_values.assign(tf.where(train_dones[i], train_rewards[i], train_rewards[i] + self.gamma * boostrapped_values))
            train_returns[i].assign(boostrapped_values)

        return tf.reshape(train_states, shape=(-1, 84, 84, 4)), tf.reshape(train_actions, shape=(-1,)), tf.reshape(train_returns, shape=(-1,))

    def compute_loss(self, action_logits: tf.Tensor, action_log_probas: tf.Tensor, state_values: tf.Tensor, returns: tf.Tensor) -> tf.Tensor:
        advantage = tf.stop_gradient(returns - state_values)
        action_loss = - tf.reduce_mean(advantage * action_log_probas)
        value_loss = self.critic_loss_fn(returns, state_values)

        policy = tf.nn.softmax(action_logits)
        log_policy = tf.nn.log_softmax(action_logits)
        entropy_loss = 0.01 * tf.reduce_mean(tf.math.reduce_sum(policy * log_policy, axis=-1))

        return action_loss + value_loss + entropy_loss, action_loss, value_loss, entropy_loss

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, 84, 84, 4), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    ])
    def train_step(self, states: tf.Tensor, actions: tf.Tensor, returns: tf.Tensor):
        returns = tf.expand_dims(returns, axis=1)
        with tf.GradientTape() as tape:
            action_logits, state_values = self.model(states)
            action_log_probas = tf.expand_dims(tf.gather(tf.nn.log_softmax(action_logits), actions, batch_dims=1), axis=1)
            loss, actor_loss, critic_loss, entropy_loss = self.compute_loss(action_logits, action_log_probas, state_values, returns)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, actor_loss, critic_loss, entropy_loss

    def learn_from_episode(self):
        start = perf_counter()
        current_states, _ = self.env.reset() if self.current_states is None else (self.current_states, None)
        current_states = tf.Variable(current_states, dtype=tf.float32)
        states, actions, returns = self.collect_rollout(current_states, self.states, self.actions, self.rewards, self.dones, self.boostrapped_values, self.train_returns)
        loss, actor_loss, critic_loss, entropy_loss = self.train_step(states, actions, returns)
        end = perf_counter()
        return np.mean(returns), loss, actor_loss, critic_loss, entropy_loss, end - start

    def test(self):
        states, _ = self.test_env.reset()
        frames = []
        total_reward = 0
        for _ in range(5_000):
            frame = self.test_env.render()
            frames.append(frame)
            states = np.transpose(states, axes=(1, 2, 0))
            states = states.astype(np.float32)[np.newaxis] / 255.0
            action_logits, _ = self.model(states)
            optimal_action = tf.argmax(action_logits, axis=-1).numpy()[0]
            states, reward, terminated, truncated, _ = self.test_env.step(optimal_action)
            total_reward += reward
            if terminated or truncated:
                break
        return total_reward, frames
        
def main():
    metrics = MetricLog()
    gym.register_envs(ale_py)
    envs = SyncVectorEnv([make_atari_env("ALE/BattleZone-v5") for _ in range(8)])
    test_env = make_atari_env("ALE/BattleZone-v5", render_mode="rgb_array")()

    model = Actor2Critic((84, 84, 4), 18)
    dummy_input = tf.zeros((1, 84, 84, 4))
    _ = model(dummy_input)
    gamma = 0.99
    critic_loss_fn = keras.losses.Huber()
    optimizer = keras.optimizers.Nadam(learning_rate=0.0015, clipnorm=0.1)

    agent = Agent(envs, test_env, model, optimizer, critic_loss_fn, gamma, 100)
    reward_threhold = 3000

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_logs_dir = "logs/A2C/BattleZone/train" + current_time
    test_logs_dir = "logs/A2C/BattleZone/test" + current_time

    train_summary_writer = tf.summary.create_file_writer(train_logs_dir)
    test_summary_writer = tf.summary.create_file_writer(test_logs_dir)
    
    moving_average_reward: deque = deque(maxlen=100)
    try:
        t = tqdm.trange(10_000)
        for iteration in t:
            average_reward, loss, actor_loss, critic_loss, entropy_loss, iteration_time = agent.learn_from_episode()
            moving_average_reward.append(average_reward)
            t.set_postfix(episode_reward=average_reward, running_reward=np.mean(moving_average_reward))

            if iteration % 500 == 0:
                test_reward, frames = agent.test()
                imageio.mimsave(f"BattleZone-{iteration}.gif", frames, fps=30)
                del frames
                gc.collect()

                metrics.log(iteration, iteration_time)

                with train_summary_writer.as_default():
                    tf.summary.scalar("train reward", average_reward, step=iteration)
                    tf.summary.scalar("actor loss", actor_loss, step=iteration)
                    tf.summary.scalar("critic loss", critic_loss, step=iteration)
                    tf.summary.scalar("entropy loss", entropy_loss, step=iteration)
                    tf.summary.scalar("loss", loss, step=iteration)
                    tf.summary.scalar("iteration_time", iteration_time, step=iteration)
                
            
                with test_summary_writer.as_default():
                    tf.summary.scalar("test reward", test_reward, step=iteration)

                if test_reward > reward_threhold:
                        print("Problem Solved!")
            
    except KeyboardInterrupt:
        pass
    finally:
        envs.close()
        test_env.close()

if __name__ == "__main__":
    main()