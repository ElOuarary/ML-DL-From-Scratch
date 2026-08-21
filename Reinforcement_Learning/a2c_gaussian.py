import argparse
from collections import deque
from datetime import datetime
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import keras
import numpy as np
import tensorflow as tf
import tqdm

from utils import MetricLog


def build_arg_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--num-envs", default=8, type=int)
    arg_parser.add_argument("--gamma", default=0.99, type=float)
    arg_parser.add_argument("--alpha", default=0.0001, type=float)
    arg_parser.add_argument("--entropy-beta", default=0.01, type=float)
    arg_parser.add_argument("--num-steps", default=5, type=int)
    arg_parser.add_argument("--train-iteration", default=1_000_000, type=int)
    arg_parser.add_argument("--log-interval", default=1_000, type=int)
    arg_parser.add_argument("--reward-threshold", default=500_000, type=float)
    arg_parser.add_argument("--test-steps", default=5_000, type=int)
    return arg_parser


@keras.saving.register_keras_serializable(
    package="a2cguassian_model", name="A2CGuassian"
)
class A2C_Guassian(keras.Model):
    def __init__(self, observation_space: tuple[int,], action_space: int, **kwargs):
        super().__init__(**kwargs)
        self.observation_space = observation_space
        self.action_space = action_space
        self.shared_network = keras.Sequential(
            [
                keras.layers.InputLayer(self.observation_space),
                keras.layers.Dense(512, activation="relu"),
                keras.layers.Dense(256, activation="relu"),
            ]
        )
        self.actor_mu = keras.layers.Dense(self.action_space, activation="tanh")
        self.actor_std = keras.layers.Dense(self.action_space, activation="softplus")
        self.critic = keras.layers.Dense(1)

    def call(self, obs: np.ndarray | tf.Tensor):
        x = self.shared_network(obs)
        return self.actor_mu(x), self.actor_std(x), self.critic(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "observation_space": self.observation_space,
                "action_space": self.action_space,
            }
        )
        return config


class Agent:
    def __init__(self, env, test_env, model, optimizer, loss_fn, gamma, n_steps, beta):
        self.env = env
        self.observation_shape = self.env.observation_space.shape[1]
        self.action_shape = self.env.action_space.shape[1]
        self.obs, _ = env.reset()
        self.test_env = test_env
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.gamma = gamma
        self.n_steps = n_steps
        self.beta = beta

    @tf.function
    def predict(self, obs: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return self.model(obs)

    def env_step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        next_obs, reward, _, truncated, _ = self.env.step(action)
        return (
            next_obs.astype(np.float64),
            reward.astype(np.float32),
            truncated.astype(np.bool),
        )

    def collect_data(
        self, obs: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        initial_shape = obs.shape

        log_probabilities = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        actions_deviation = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        state_values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        dones = tf.TensorArray(dtype=tf.bool, size=0, dynamic_size=True)

        for i in tf.range(self.n_steps):
            mu, std, state_value = self.predict(obs)
            actions = tf.random.normal(
                tf.shape(mu), mean=mu, stddev=std, dtype=tf.float32
            )
            log_probability = -(
                tf.math.pow(actions - mu, 2) / (2 * std**2 + 1e-5)
                + 0.5 * tf.math.log(2 * np.pi * (std + 1e-5) ** 2)
            )
            actions = tf.clip_by_value(actions, clip_value_min=-0.4, clip_value_max=0.4)
            obs, reward, done = tf.numpy_function(
                self.env_step,
                inp=[tf.cast(actions, dtype=tf.float32)],
                Tout=[tf.float64, tf.float32, tf.bool],
            )

            log_probabilities = log_probabilities.write(i, log_probability)
            actions_deviation = actions_deviation.write(i, std + 1e-5)
            state_values = state_values.write(i, state_value)
            rewards = rewards.write(i, reward)
            dones = dones.write(i, done)

            obs.set_shape(initial_shape)

        log_probabilities = log_probabilities.stack()
        actions_deviation = actions_deviation.stack()
        state_values = state_values.stack()
        rewards = rewards.stack()
        dones = dones.stack()

        return (
            tf.reshape(log_probabilities, shape=(-1, self.action_shape)),
            tf.reshape(state_values, shape=(-1, 1)),
            rewards,
            dones,
            tf.reshape(actions_deviation, shape=(-1, self.action_shape)),
            obs,
        )

    def compute_boostrapped_returns(
        self, next_obs: tf.Tensor, rewards: tf.Tensor, dones: tf.Tensor
    ) -> tf.Tensor:
        _, _, next_state_value = self.model(next_obs)
        boostrapped_value = next_state_value[:, 0]
        boostrapped_shape = boostrapped_value.shape

        returns = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        for i in tf.reverse(tf.range(self.n_steps), axis=[-1]):
            boostrapped_value = tf.where(
                dones[i], rewards[i], rewards[i] + self.gamma * boostrapped_value
            )
            returns = returns.write(i, boostrapped_value)
            boostrapped_value.set_shape(boostrapped_shape)

        returns = returns.stack()

        return tf.reshape(returns, shape=(-1, 1))

    def compute_loss(
        self,
        log_probability: tf.Tensor,
        state_value: tf.Tensor,
        reward: tf.Tensor,
        action_deviation: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        advantage = tf.stop_gradient(reward - state_value)
        advantage = (advantage - tf.reduce_mean(advantage)) / (
            tf.math.reduce_std(advantage) + 1e-8
        )
        policy_loss = tf.reduce_mean(
            advantage * tf.reduce_sum(log_probability, axis=-1, keepdims=True)
        )
        value_loss = self.loss_fn(state_value, reward)
        entropy_loss = self.beta * tf.reduce_mean(
            0.5 * tf.math.log(2 * np.pi * np.e * action_deviation**2)
        )
        return (
            policy_loss + value_loss - entropy_loss,
            policy_loss,
            value_loss,
            entropy_loss,
        )

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 348), dtype=tf.float64)])
    def train_step(
        self, obs: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            (
                log_probabilities,
                state_values,
                rewards,
                dones,
                actions_deviation,
                next_obs,
            ) = self.collect_data(obs)
            returns = tf.stop_gradient(
                self.compute_boostrapped_returns(next_obs, rewards, dones)
            )
            loss, policy_loss, value_loss, entropy_loss = self.compute_loss(
                log_probabilities, state_values, returns, actions_deviation
            )

        grad = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
        return (
            tf.reduce_mean(tf.reduce_sum(rewards, axis=0)),
            loss,
            policy_loss,
            value_loss,
            entropy_loss,
            next_obs,
        )

    def learn(self):
        self.obs, _ = self.env.reset() if self.obs is None else (self.obs, None)
        rewards, loss, policy_loss, value_loss, entropy_loss, next_obs = (
            self.train_step(self.obs)
        )
        self.obs = next_obs.numpy()
        return rewards, loss, policy_loss, value_loss, entropy_loss

    def test(self) -> np.ndarray:
        obs, _ = self.test_env.reset()
        active = np.full(self.test_env.num_envs, np.True_)
        total_reward = np.zeros(self.test_env.num_envs)

        while np.any(active):
            actions, _, _ = self.model(obs)
            actions = actions[:].numpy()
            clipped_actions = np.clip(actions, -0.4, 0.4)
            obs, reward, _, truncated, _ = self.test_env.step(clipped_actions)
            total_reward += reward * active
            active = np.logical_and(active, np.logical_not(truncated))

        episode_reward = np.mean(total_reward)
        return episode_reward


def main():
    envs, test_envs = None, None

    arg_parser = build_arg_parser()
    args = arg_parser.parse_args()

    NUM_ENVS = args.num_envs
    GAMMA = args.gamma
    ALPHA = args.alpha
    ENTROPY_BETA = args.entropy_beta
    NUM_STEPS = args.num_steps
    TRAIN_ITERATION = args.train_iteration
    REWARD_THRESHOLD = args.reward_threshold
    TEST_STEPS = args.test_steps

    try:
        metrics = MetricLog("humanoidStandup.csv")
        envs = gym.make_vec(
            "HumanoidStandup-v5", num_envs=NUM_ENVS, vectorization_mode="async"
        )
        test_envs = gym.make_vec(
            "HumanoidStandup-v5", num_envs=10, vectorization_mode="async"
        )

        obs = tf.random.normal((1, envs.observation_space.shape[1]))
        model = A2C_Guassian(
            (envs.observation_space.shape[1],), envs.action_space.shape[1]
        )
        _ = model(obs)

        if Path("a2c_guassian.weights.h5").exists():
            model.load_weights("a2c_guassian.weights.h5")

        value_loss_fn = keras.losses.MeanSquaredError()
        optimizer = keras.optimizers.Adam(learning_rate=ALPHA, clipnorm=0.1)

        agent = Agent(
            envs,
            test_envs,
            model,
            optimizer,
            value_loss_fn,
            GAMMA,
            NUM_STEPS,
            ENTROPY_BETA,
        )

        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        train_logs_dir = "logs/A2C_Guassian/train/" + current_time
        test_logs_dir = "logs/A2C_Guassian/test/" + current_time

        train_file_writer = tf.summary.create_file_writer(train_logs_dir)
        test_file_writer = tf.summary.create_file_writer(test_logs_dir)

        moving_average_reward: deque = deque(maxlen=500)
        t = tqdm.trange(TRAIN_ITERATION)
        for iteration in t:
            start = perf_counter()
            train_reward, loss, policy_loss, value_loss, entropy_loss = agent.learn()
            end = perf_counter() - start

            moving_average_reward.append(train_reward)
            running_reward = np.mean(moving_average_reward)

            t.set_postfix(
                episode_reward=train_reward.numpy(), running_reward=running_reward
            )

            if iteration % TEST_STEPS == 0:
                episode_reward = agent.test()
                metrics.log(iteration, end)

                with train_file_writer.as_default():
                    tf.summary.scalar("train reward", running_reward, step=iteration)
                    tf.summary.scalar("loss", loss, step=iteration)
                    tf.summary.scalar("policy loss", policy_loss, step=iteration)
                    tf.summary.scalar("value loss", value_loss, step=iteration)
                    tf.summary.scalar("entropy loss", entropy_loss, step=iteration)

                with test_file_writer.as_default():
                    tf.summary.scalar("test reward", episode_reward, step=iteration)

                model.save_weights("a2c_guassian.weights.h5")

                if episode_reward > REWARD_THRESHOLD:
                    model.save("a2c_guassian.keras")
                    print("Problem Solved")

    except KeyboardInterrupt:
        pass
    finally:
        train_file_writer.close()
        test_file_writer.close()
        if envs is not None:
            envs.close()
        if test_envs is not None:
            test_envs.close()


if __name__ == "__main__":
    main()
