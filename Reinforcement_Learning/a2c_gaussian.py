from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
import tensorflow as tf
import tqdm
from tensorflow import keras

class A2C_Guassian(keras.Model):
    def __init__(self, observation_space: tuple[int,], action_space: int, **kwargs):
        super().__init__(**kwargs)
        self.observation_space = observation_space
        self.action_space = action_space
        self.shared_network = keras.Sequential([
            keras.layers.InputLayer(self.observation_space),
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dense(256, activation="relu")
        ])
        self.actor_mu = keras.layers.Dense(self.action_space, activation="tanh")
        self.actor_std = keras.layers.Dense(self.action_space, activation="softmax")
        self.critic = keras.layers.Dense(1)

    def call(self, obs):
        x = self.shared_network(obs)
        return self.actor_mu(x), self.actor_std(x), self.critic(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "observation_space": self.observation_space,
            "action_space": self.action_space
        })
        return config

class Agent:
    def __init__(self, env, test_env, demo_env, model, optimizer, loss_fn, gamma, n_steps, beta):
        self.env = env
        self.observation_shape = self.env.observation_space.shape[1]
        self.action_shape = self.env.action_space.shape[1]
        self.obs, _ = env.reset()
        self.test_env = test_env
        self.demo_env = demo_env
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.gamma = gamma
        self.n_steps = n_steps
        self.beta = beta

    @tf.function
    def predict(self, obs: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return self.model(obs)
    
    def env_step(self, action: np.ndarray):
        next_obs, reward, _, truncated, _ = self.env.step(action)
        return next_obs.astype(np.float64), reward.astype(np.float32), truncated.astype(np.bool)

    # Need to be decomposed into smaller function especially for computing rollout
    def collect_data(self, obs=tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        initial_shape = obs.shape

        log_probabilities = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        actions_deviation = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        state_values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        dones = tf.TensorArray(dtype=tf.bool, size=0, dynamic_size=True)

        for i in tf.range(self.n_steps):
            mu, std, state_value = self.predict(obs)
            actions = tf.random.normal(self.env.action_space.shape, mean=mu, stddev=std, dtype=tf.float32)
            actions = tf.clip_by_value(actions, clip_value_min=-0.4, clip_value_max=0.4)
            obs, reward, done = tf.numpy_function(self.env_step, inp=[tf.cast(actions, dtype=tf.float32)], Tout=[tf.float64, tf.float32, tf.bool])

            log_probability = tf.math.pow(actions - mu, 2) / 2 * (std **2 + 1e-5) + 0.5 * tf.math.log(2 * np.pi * (std + 1e-5) ** 2)
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

        _, _, next_state_value = self.model(obs)
        boostraped_value = next_state_value[:, 0]
        boostrapped_shape = boostraped_value.shape

        returns = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        for i in tf.reverse(tf.range(self.n_steps), axis=[0]):
            boostraped_value = tf.where(dones[i], rewards[i], rewards[i] + self.gamma * boostraped_value)
            returns = returns.write(i, boostraped_value)
            boostraped_value.set_shape(boostrapped_shape)
        returns = returns.stack()

        return tf.reshape(log_probabilities, shape=(-1, self.action_shape)), tf.reshape(state_values, shape=(-1, 1)), tf.reshape(rewards, shape=(-1, 1)), tf.reshape(actions_deviation, shape=(-1, self.action_shape))

    def compute_loss(self, log_probability, state_value, reward, action_deviation):
        advantage = tf.stop_gradient((reward - state_value))
        advantage = advantage - tf.reduce_mean(advantage) / (tf.math.reduce_std(advantage) + 1e-8)
        policy_loss = tf.reduce_mean(advantage * log_probability)
        value_loss = self.loss_fn(state_value, reward)
        entropy_loss = tf.reduce_mean(0.5 * tf.math.log(2 * np.pi * action_deviation ** 2))
        return policy_loss + value_loss + entropy_loss, policy_loss, value_loss, entropy_loss

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 348), dtype=tf.float64)])
    def train_step(self, obs: tf.Tensor):
        with tf.GradientTape() as tape:
            log_probabilities, state_values, rewards, actions_deviation = self.collect_data(obs)
            loss, policy_loss, value_loss, entropy_loss = self.compute_loss(log_probabilities, state_values, rewards, actions_deviation)

        grad = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
        return tf.reduce_mean(tf.reduce_sum(rewards, axis=-1)), loss, policy_loss, value_loss, entropy_loss

    def learn(self):
        obs, _ = self.env.reset() if self.obs is None else (self.obs, None)
        return self.train_step(obs)

    def test(self, demo=False):
        obs, _ = self.test_env.reset() if not demo else self.demo_env.reset()
        total_reward = 0
        frames = []
        while True:
            if demo:
                obs = obs[np.newaxis]
                frame = self.demo_env.render()
                frames.append(frame)
            actions, _, _ = self.model(obs)
            actions = actions[:].numpy()
            clipped_actions = np.clip(actions, -0.4, 0.4)
            obs, reward, _, truncated, _ = self.test_env.step(actions) if not demo else self.demo_env.step(clipped_actions[0])
            total_reward += reward
            if np.any(truncated):
                break

        episode_reward = np.mean(np.sum(total_reward, axis=-1)) if not demo else np.sum(total_reward)
        return episode_reward, frames

def main():
    envs, test_envs, demo_env = None, None, None

    try:
        envs = gym.make_vec("HumanoidStandup-v5", num_envs=8, vectorization_mode="async")
        test_envs = gym.make_vec("HumanoidStandup-v5", num_envs=10, vectorization_mode="async")
        demo_env = gym.make("HumanoidStandup-v5", render_mode="rgb_array")
        
        obs = tf.random.normal((1, envs.observation_space.shape[1]))
        model = A2C_Guassian((envs.observation_space.shape[1],), envs.action_space.shape[1])
        _ = model(obs)

        if Path("a2c_guassian.weights.h5").exists():
            model.load_weights("a2c_guassian.weights.h5")

        value_loss_fn = keras.losses.MeanSquaredError()
        optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=0.1)

        agent = Agent(envs, test_envs, demo_env, model, optimizer, value_loss_fn, 0.99, 50, 0.01)

        t = tqdm.trange(10_000)
        for iteration in t:
            train_reward, loss, policy_loss, value_loss, entropy_loss = agent.learn()
            print(train_reward)

            if iteration % 1000:
                episode_reward, frames = agent.test()
                print(episode_reward)

                model.save_weights("a2c_guassian.weights.h5")

                if episode_reward > 1000:
                    model.save("a2c_guassian.keras")

                    episode_reward, frames = agent.test(demo=True)
                    imageio.mimsave("HumanoidStandup/a2c/demo-episode-{iteration}.gif", frames, fps=30)


    except KeyboardInterrupt:
        pass
    finally:
        if envs is not None:
            envs.close() 
        if test_envs is not None:
            test_envs.close()
        if demo_env is not None:
            demo_env.close()

if __name__ == "__main__":
    main()