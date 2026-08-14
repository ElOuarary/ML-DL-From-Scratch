import argparse
import gc
from collections import deque
from datetime import datetime
from time import perf_counter

import ale_py
import gymnasium as gym
import imageio
import numpy as np
import tensorflow as tf
import tqdm
from gymnasium.vector import SyncVectorEnv
from tensorflow import keras

from utils import MetricLog, make_atari_env


def build_arg_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--num-envs", default=8, type=int)
    arg_parser.add_argument("--gamma", default=0.99, type=float)
    arg_parser.add_argument("--alpha", default=0.001, type=float)
    arg_parser.add_argument("--entropy-beta", default=0.01, type=float)
    arg_parser.add_argument("--num-steps", default=5, type=int)
    arg_parser.add_argument("--reward-threshold", default=5_000, type=float)
    return arg_parser


class Actor2Critic(keras.Model):
    def __init__(self, observation_space: int, action_space: int):
        super().__init__()
        self.shared_network = keras.models.Sequential(
            [
                keras.layers.InputLayer(observation_space),
                keras.layers.Conv2D(32, kernel_size=8, strides=4, activation="relu"),
                keras.layers.Conv2D(64, kernel_size=4, strides=2, activation="relu"),
                keras.layers.Conv2D(64, kernel_size=3, strides=1, activation="relu"),
                keras.layers.Flatten(),
                keras.layers.Dense(512, activation="relu"),
            ]
        )
        self.actor = keras.layers.Dense(action_space)
        self.critic = keras.layers.Dense(1)

    def call(self, obs: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x = self.shared_network(obs)
        return self.actor(x), self.critic(x)


class Agent:
    def __init__(
        self, env, test_env, model, optimizer, critic_loss_fn, gamma, beta, n_steps
    ):
        self.env = env
        self.current_states, _ = env.reset()
        self.obs_shape = (
            self.env.num_envs,
            self.env.observation_space.shape[2],
            self.env.observation_space.shape[3],
            self.env.observation_space.shape[1],
        )
        self.test_env = test_env
        self.model = model
        self.critic_loss_fn = critic_loss_fn
        self.optimizer = optimizer
        self.gamma = gamma
        self.beta = beta
        self.n_steps = n_steps

    @tf.function
    def predict(self, state):
        action_logits, _ = self.model(state)
        action = tf.random.categorical(action_logits, num_samples=1, dtype=tf.int32)
        return action

    def collect_rollout(
        self, states: np.ndarray
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        train_states = np.zeros(shape=(self.n_steps, *self.obs_shape), dtype=np.float32)
        train_actions = np.zeros(
            shape=(self.n_steps, self.env.num_envs), dtype=np.int32
        )
        train_rewards = np.zeros(
            shape=(self.n_steps, self.env.num_envs), dtype=np.float32
        )
        train_dones = np.zeros(shape=(self.n_steps, self.env.num_envs), dtype=np.bool)

        for step in range(self.n_steps):
            states_transpose = (
                np.transpose(states.astype(np.float32), axes=[0, 2, 3, 1]) / 255.0
            )
            actions = self.predict(states_transpose).numpy().flatten()
            next_states, rewards, terminated, _, _ = self.env.step(actions)

            train_states[step] = states_transpose
            train_actions[step] = actions
            train_rewards[step] = rewards
            train_dones[step] = terminated

            states = next_states

        self.current_states = states

        next_states = (
            np.transpose(next_states.astype(np.float32), axes=(0, 2, 3, 1)) / 255.0
        )
        _, next_state_value = self.model(next_states)
        boostrapped_values = next_state_value[:, 0].numpy()

        train_returns = np.zeros_like(train_rewards)
        for i in reversed(range(self.n_steps)):
            boostrapped_values = np.where(
                train_dones[i],
                train_rewards[i],
                train_rewards[i] + self.gamma * boostrapped_values,
            )
            train_returns[i] = boostrapped_values

        return (
            tf.convert_to_tensor(train_states.reshape(-1, 84, 84, 4), dtype=tf.float32),
            tf.convert_to_tensor(
                train_actions.reshape(
                    -1,
                ),
                dtype=tf.int32,
            ),
            tf.convert_to_tensor(
                train_returns.reshape(
                    -1,
                ),
                dtype=tf.float32,
            ),
        )

    def compute_loss(
        self,
        action_logits: tf.Tensor,
        action_log_probas: tf.Tensor,
        state_values: tf.Tensor,
        returns: tf.Tensor,
    ) -> tf.Tensor:
        advantage = tf.stop_gradient(returns - state_values)
        normalized_advantage = (advantage - tf.reduce_mean(advantage)) / (
            tf.math.reduce_std(advantage) + 1e-8
        )
        action_loss = -tf.reduce_mean(normalized_advantage * action_log_probas)
        value_loss = self.critic_loss_fn(returns, state_values)

        policy = tf.nn.softmax(action_logits)
        log_policy = tf.nn.log_softmax(action_logits)
        entropy = tf.reduce_mean(tf.math.reduce_sum(policy * log_policy, axis=-1))

        return (
            action_loss + value_loss + self.beta * entropy,
            action_loss,
            value_loss,
            entropy,
        )

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, 84, 84, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        ],
        jit_compile=True,
    )
    def train_step(
        self, states: tf.Tensor, actions: tf.Tensor, returns: tf.Tensor
    ) -> tuple(tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor):
        returns = tf.expand_dims(returns, axis=1)
        with tf.GradientTape() as tape:
            action_logits, state_values = self.model(states)
            action_log_probas = tf.expand_dims(
                tf.gather(tf.nn.log_softmax(action_logits), actions, batch_dims=1),
                axis=1,
            )
            loss, actor_loss, critic_loss, entropy_loss = self.compute_loss(
                action_logits, action_log_probas, state_values, returns
            )

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, actor_loss, critic_loss, entropy_loss

    def learn_from_episode(self):
        start = perf_counter()
        current_states, _ = (
            self.env.reset()
            if self.current_states is None
            else (self.current_states, None)
        )
        states, actions, returns = self.collect_rollout(current_states)
        loss, actor_loss, critic_loss, entropy_loss = self.train_step(
            states, actions, returns
        )
        end = perf_counter()
        return (
            np.mean(returns),
            loss,
            actor_loss,
            critic_loss,
            entropy_loss,
            end - start,
        )

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
            states, reward, terminated, truncated, _ = self.test_env.step(
                optimal_action
            )
            total_reward += reward
            if terminated or truncated:
                break
        return total_reward, frames


def main():
    metrics = MetricLog()
    gym.register_envs(ale_py)

    args = build_arg_parser().parse_args()
    NUM_ENVS = args.num_envs
    GAMMA = args.gamma
    ALPHA = args.alpha
    ENTROPY_BETA = args.entropy_beta
    NUM_STEPS = args.num_steps
    REWARD_THRESHOLD = args.reward_threshold

    envs = SyncVectorEnv([make_atari_env("ALE/BattleZone-v5") for _ in range(NUM_ENVS)])
    test_env = make_atari_env("ALE/BattleZone-v5", render_mode="rgb_array")()

    model = Actor2Critic((84, 84, 4), 18)
    dummy_input = tf.zeros((1, 84, 84, 4))
    _ = model(dummy_input)

    critic_loss_fn = keras.losses.Huber()
    optimizer = keras.optimizers.Nadam(learning_rate=ALPHA, clipnorm=0.1)

    agent = Agent(
        envs, test_env, model, optimizer, critic_loss_fn, GAMMA, ENTROPY_BETA, NUM_STEPS
    )

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_logs_dir = "logs/A2C/BattleZone/train" + current_time
    test_logs_dir = "logs/A2C/BattleZone/test" + current_time

    train_summary_writer = tf.summary.create_file_writer(train_logs_dir)
    test_summary_writer = tf.summary.create_file_writer(test_logs_dir)

    moving_average_reward: deque = deque(maxlen=500)
    try:
        t = tqdm.trange(100_000)
        for iteration in t:
            (
                average_reward,
                loss,
                actor_loss,
                critic_loss,
                entropy_loss,
                iteration_time,
            ) = agent.learn_from_episode()
            moving_average_reward.append(average_reward)
            t.set_postfix(
                episode_reward=average_reward,
                running_reward=np.mean(moving_average_reward),
            )

            if iteration % 1000 == 0:
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

                if test_reward > REWARD_THRESHOLD:
                    print("Problem Solved!")

    except KeyboardInterrupt:
        pass
    finally:
        envs.close()
        test_env.close()
        model.save("a2c.keras")


if __name__ == "__main__":
    main()
