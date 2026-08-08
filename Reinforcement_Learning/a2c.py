import gymnasium as gym
# from gymnasium.wrappers import RecordVideo
import ale_py
import numpy as np
from scipy.signal import lfilter
import tensorflow as tf
from tensorflow import keras

import imageio
from datetime import datetime
import psutil, os, csv
from time import perf_counter
from typing import Tuple

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
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(64, kernel_size=4, strides=2, activation="relu"),
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(64, kernel_size=3, strides=1, activation="relu"),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation="relu")
        ])
        self.actor = keras.layers.Dense(action_space)
        self.critic = keras.layers.Dense(1)

    def call(self, obs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.shared_network(obs)
        return self.actor(x), self.critic(x)

class Agent:
    def __init__(self, env, test_env, model, optimizer, critic_loss_fn, gamma):
        self.env = env
        self.test_env = test_env
        self.model = model
        self.critic_loss_fn = critic_loss_fn
        self.optimizer = optimizer
        self.gamma = gamma

    @tf.function
    def predict(self, state):
        action_logits, _ = self.model(state)
        action = tf.random.categorical(action_logits, num_samples=1)[0, 0]
        return action

    def collect_episode(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        states, actions, rewards = [], [], []
        state, _ = self.env.reset()
        for _ in range(1000):
            state = state.astype(np.float32)[np.newaxis] / 255.0
            action = self.predict(state)
            next_state, reward, terminated, truncated, _  = self.env.step(action.numpy())

            states.append(state[0])
            actions.append(action)
            rewards.append(reward)
            
            if terminated or truncated:
                state, _ = self.env.reset()

            state = next_state

        return np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.int32), np.asarray(rewards, dtype=np.float32)

    def compute_returns(self, episode_rewards: tf.Tensor) -> tf.Tensor:
        n = tf.shape(episode_rewards)[0]
        discount = 0.99 ** tf.cast(tf.range(n), dtype=tf.float32)
        returns = tf.cumsum(tf.reverse(episode_rewards * discount, axis=[0]))
        mean = tf.reduce_mean(returns)
        std = tf.math.reduce_std(returns)
        return (returns - mean) / (std + 1e-8)
    
    def compute_loss(self, action_probas: tf.Tensor, state_values: tf.Tensor, returns: tf.Tensor) -> tf.Tensor:
        action_loss = - tf.reduce_mean((returns - state_values) * action_probas)
        value_loss = self.critic_loss_fn(returns, state_values)
        return action_loss + value_loss, action_loss, value_loss

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 210, 160, 3), dtype=tf.float32), tf.TensorSpec(shape=(1000,), dtype=tf.int32), tf.TensorSpec(shape=(1000,), dtype=tf.float32)])
    def train_step(self, states: tf.Tensor, actions: tf.Tensor, rewards: tf.Tensor):
        discounted_rewards = self.compute_returns(rewards)
        with tf.GradientTape() as tape:
            action_logits, state_values = self.model(states)
            action_probas = tf.gather(tf.nn.log_softmax(action_logits), actions, batch_dims=1)
            loss, actor_loss, critic_loss = self.compute_loss(action_probas, state_values, discounted_rewards)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return tf.reduce_sum(rewards), loss, actor_loss, critic_loss

    def learn_from_episode(self):
        states, actions, rewards = self.collect_episode()
        start = perf_counter()
        episode_reward, loss, actor_loss, critic_loss = self.train_step(states, actions, rewards)
        end = perf_counter()
        return episode_reward, loss, actor_loss, critic_loss, end - start

    def test(self):
        states, _ = self.test_env.reset()
        frames = []
        total_reward = 0
        while True:
            frame = self.test_env.render()
            frames.append(frame)
            states = states.astype(np.float32)[np.newaxis] / 255.0
            action_logits, _ = self.model(states)
            action_probas = tf.nn.softmax(action_logits)
            optimal_action = tf.argmax(action_probas, axis=-1).numpy()[0]
            states, reward, terminated, truncated, _ = self.test_env.step(optimal_action)
            total_reward += reward
            if terminated or truncated:
                break
        return total_reward, frames
        
def main():
    metrics = MetricLog()
    gym.register_envs(ale_py)
    env = gym.make("ALE/BattleZone-v5")
    test_env = gym.make("ALE/BattleZone-v5", render_mode="rgb_array")
    model = Actor2Critic(env.observation_space.shape, 18)
    gamma = 0.99

    critic_loss_fn = keras.losses.Huber()
    optimizer = keras.optimizers.Nadam(global_clipnorm=0.5)

    agent = Agent(env, test_env, model, optimizer, critic_loss_fn, gamma)
    reward_threhold = 3000

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_logs_dir = "logs/A2C/BattleZone/train" + current_time
    test_logs_dir = "logs/A2C/BattleZone/test" + current_time

    train_summary_writer = tf.summary.create_file_writer(train_logs_dir)
    test_summary_writer = tf.summary.create_file_writer(test_logs_dir)

    iteration = 0
    try:
        while True:
            start = perf_counter()
            episode_reward, loss, actor_loss, critic_loss, iteration_time = agent.learn_from_episode()
            iteration_step = perf_counter() - start
            metrics.log(iteration, iteration_step)
            with train_summary_writer.as_default():
                tf.summary.scalar("train reward", episode_reward, step=iteration)
                tf.summary.scalar("actor loss", actor_loss, step=iteration)
                tf.summary.scalar("critic loss", critic_loss, step=iteration)
                tf.summary.scalar("loss", loss, step=iteration)
                tf.summary.scalar("iteration_time", iteration_time, step=iteration)

            if iteration % 500 == 0:
                test_reward , frames = agent.test()
                imageio.mimsave(f"BattleZone-{iteration}.gif", frames, fps=30)
            
                with test_summary_writer.as_default():
                    tf.summary.scalar("test reward", test_reward, step=iteration)
                    print(f"Iteration {iteration} - Average Reward: {test_reward}")

                if test_reward > reward_threhold:
                        print("Problem Solved!")
            
            iteration += 1
            
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        test_env.close()

if __name__ == "__main__":
    main()