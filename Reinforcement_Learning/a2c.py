import gymnasium as gym
# from gymnasium.wrappers import RecordVideo
import ale_py
import numpy as np
import tensorflow as tf
from tensorflow import keras

import imageio
from datetime import datetime
from time import perf_counter
from typing import Tuple

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
            keras.layers.Dense(256, activation="relu"),
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

    def env_step(self, action: tf.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        def _env_step(action):
            next_state, reward, termiated, truncated, _ = self.env.step(action)
            return next_state.astype(np.float32), np.array(reward, np.float32), np.array(termiated | truncated, np.bool)
        return tf.numpy_function(
            func=_env_step,
            inp=[action],
            Tout=[tf.float32, tf.float32, tf.bool]
        )

    def play_episode(self, initial_state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        state = initial_state
        initial_shape = initial_state.shape
        action_probas = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        state_values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        iteration = 0
        while tf.constant(True):
            state = tf.expand_dims(state, axis=0)
            action_logits, state_value = self.model(state)
            action = tf.random.categorical(action_logits, 1)[0, 0]
            action_proba = tf.nn.log_softmax(action_logits)
            state, reward, done = self.env_step(action)

            action_probas = action_probas.write(iteration, action_proba[0, action])
            state_values = state_values.write(iteration, tf.squeeze(state_value))
            rewards = rewards.write(iteration, reward)

            state.set_shape(initial_shape)

            iteration += 1
 
            if done:
                break

        action_probas = tf.expand_dims(action_probas.stack(), axis=0)
        state_values = tf.expand_dims(state_values.stack(), axis=0)
        rewards = rewards.stack()
        
        return action_probas, state_values, rewards


    def compute_returns(self, episode_rewards: tf.Tensor, gamma: float) -> tf.Tensor:
        episode_rewards = tf.ensure_shape(episode_rewards, [None])
        reversed_rewards = tf.reverse(episode_rewards, axis=[0])
        discounted_reversed = tf.scan(
            fn=lambda acc, r: r + gamma * acc,
            elems=reversed_rewards,
            initializer=tf.constant(0, dtype=tf.float32)
        )
        discounted = tf.reverse(discounted_reversed, axis=[0])
        mean = tf.reduce_mean(discounted)
        std = tf.math.reduce_std(discounted)
        return (discounted - mean) / (std + 1e-8)
    
    def compute_loss(self, action_probas: tf.Tensor, state_values: tf.Tensor, returns: tf.Tensor) -> tf.Tensor:
        action_loss = - tf.reduce_mean((state_values - returns) * action_probas)
        value_loss = self.critic_loss_fn(returns, state_values)
        return action_loss + value_loss, action_loss, value_loss

    @tf.function
    def train_step(self, initial_state: tf.Tensor):
        with tf.GradientTape() as tape:
            action_probas, state_values, rewards = self.play_episode(initial_state)
            returns = self.compute_returns(rewards, self.gamma)
            loss, action_loss, value_loss = self.compute_loss(action_probas, state_values, returns)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return tf.reduce_sum(rewards), loss, action_loss, value_loss

    def learn_from_episode(self):
        initial_state, _ = self.env.reset()
        initial_state = tf.constant(initial_state, dtype=tf.float32)
        start = perf_counter()
        episode_reward, loss, actor_loss, critic_loss = self.train_step(initial_state)
        end = perf_counter()
        return episode_reward, loss, actor_loss, critic_loss, end - start

    def test(self):
        states, _ = self.test_env.reset()
        frames = []
        total_reward = 0
        while True:
            frame = self.test_env.render()
            frames.append(frame)
            states = tf.constant(states[np.newaxis])
            action_logits, _ = self.model(states)
            action_probas = tf.nn.softmax(action_logits)
            optimal_action = tf.argmax(action_probas, axis=-1).numpy()[0]
            states, reward, terminated, truncated, _ = self.test_env.step(optimal_action)
            total_reward += reward
            if terminated or truncated:
                break
        return total_reward, frames
        
def main():
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
            episode_reward, loss, actor_loss, critic_loss, iteration_time = agent.learn_from_episode()

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