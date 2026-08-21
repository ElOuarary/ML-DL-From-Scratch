import argparse
from collections import deque, namedtuple
from datetime import UTC, datetime

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

Step = namedtuple(
    "Step", field_names=("state", "action", "reward", "next_state", "continue_mask")
)


class Agent:
    def __init__(
        self, env, epsilon, gamma, net, tg_net, loss_fn, optimizer, n_steps,buffer_size
    ):
        self.env = env
        self.state, _ = self.env.reset()
        self.action_space = env.action_space.n
        self.epsilon = epsilon
        self.gamma = gamma
        self.net = net
        self.tg_net = tg_net
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.n_steps = n_steps
        self.replay_buffer = deque(maxlen=buffer_size)

    def explore(self):
        if np.random.sample() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = self.greedy_policy(self.state)
            action = action.numpy()[0]
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        self.replay_buffer.append(
            Step(
                state=self.state,
                action=action,
                reward=reward,
                next_state=next_state,
                continue_mask=0 if terminated or truncated else 1,
            )
        )
        if terminated or truncated:
            self.state, _ = self.env.reset()
        else:
            self.state = next_state

    def compute_returns(self, indices, rewards, continues):
        returns = []
        next_state = []
        continue_mask = []
        for i, reward, continue_ in zip(indices, rewards, continues):
            if continue_:
                for j in range(i+1, min(i+self.n_steps, len(self.replay_buffer))):
                    reward += self.gamma ** (j-i) * self.replay_buffer[j].reward
                    if not self.replay_buffer[j].continue_mask:
                        break
            returns.append(reward)
        return returns

    def sample_batch(self, batch_size):
        indices = np.random.choice(
            len(self.replay_buffer), batch_size, replace=False
        )
        states, actions, rewards, next_states, continue_mask = zip(
            *[self.replay_buffer[idx] for idx in indices]
        )
        retruns = self.compute_returns(indices, rewards, continue_mask)
        states, actions, retruns, next_states, continue_mask = (
            tf.constant(states),
            tf.constant(actions),
            tf.constant(rewards, dtype=tf.float32),
            tf.constant(next_states),
            tf.constant(continue_mask, dtype=tf.float32),
        )
        return states, actions, retruns, next_states, continue_mask

    @tf.function
    def greedy_policy(self, state):
        return tf.argmax(self.net(state[np.newaxis]), axis=-1)

    @tf.function
    def compute_loss(self, state, reward, action, next_state, continue_mask):
        next_state_value = tf.reduce_max(self.tg_net(next_state), axis=-1)

        q_value_target = reward + self.gamma ** self.n_steps * continue_mask * next_state_value
        mask = tf.one_hot(action, self.action_space)
        with tf.GradientTape() as tape:
            q_value = self.net(state)
            q_value_masked = tf.reduce_sum(mask * q_value, axis=-1)
            loss = self.loss_fn(q_value_target, q_value_masked)
        gradients = tape.gradient(loss, self.net.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.net.trainable_variables))
        return loss, gradients

    @tf.function
    def compute_batch(self, states):
        return self.net(states)

    def train_model(self, batch_size):
        states, actions, rewards, next_states, continue_mask = self.sample_batch(
            batch_size
        )
        loss = self.compute_loss(states, rewards, actions, next_states, continue_mask)
        return loss

    def test(self, test_env):
        states, _ = test_env.reset()
        active = np.ones(test_env.num_envs)
        total_reward = np.zeros(test_env.num_envs)
        while np.any(active):
            q_values = self.compute_batch(states)
            action = tf.argmax(q_values, axis=-1).numpy()
            next_states, reward, terminated, truncated, _ = test_env.step(action)
            total_reward += reward * active
            active = np.logical_and(active, np.logical_not(terminated | truncated))
            states = next_states
        return total_reward.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.0005)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer-size", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--test-steps", type=int, default=500)
    parser.add_argument("--network-update", type=int, default=1000)
    parser.add_argument("--reward-target", type=int, default=200)
    parser.add_argument("--train-iteration", type=int, default=100_000)
    args = parser.parse_args()
    alpha = args.alpha
    batch_size = args.batch_size
    gamma = args.gamma
    buffer_size = args.buffer_size
    warmup_steps = args.warmup_steps
    WARMUP = min(batch_size, warmup_steps)
    test_steps = args.test_steps
    update_steps = args.network_update
    reward_target = args.reward_target
    train_iteration = args.train_iteration

    env = gym.make("LunarLander-v3")
    test_env = gym.make_vec("LunarLander-v3", num_envs=20)
    model = keras.Sequential(
        [
            keras.layers.InputLayer(env.observation_space.shape),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dense(4),
        ]
    )
    tg_model = keras.models.clone_model(model)
    tg_model.set_weights(model.get_weights())

    loss_fn = keras.losses.Huber()
    optimizer = keras.optimizers.Nadam(learning_rate=alpha, clipnorm=1)
    agent = Agent(env, 1, gamma, model, tg_model, loss_fn, optimizer, buffer_size)

    current_time = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    train_logs_dir = "logs/dqn/LunarLander/train/" + current_time
    test_logs_dir = "logs/dqn/LunarLander/test/" + current_time

    train_summary_writer = tf.summary.create_file_writer(train_logs_dir)
    test_summary_writer = tf.summary.create_file_writer(test_logs_dir)

    try:
        for i in range(1, train_iteration + 1):
            agent.explore()
            agent.epsilon = max(1 - i / 50_000, 0.01)

            if len(agent.replay_buffer) >= WARMUP:
                loss, gradients = agent.train_model(batch_size)

                if i % test_steps == 0:
                    gradient_mean = tf.reduce_mean(tf.concat([tf.reshape(g, [-1]) for g in gradients], axis=0))
                    mean_reward = agent.test(test_env)

                    with train_summary_writer.as_default():
                        tf.summary.scalar("train_loss", loss, step=i)
                        tf.summary.scalar("gradient_mean", gradient_mean, step=i)

                    with test_summary_writer.as_default():
                        tf.summary.scalar("test_mean_reward", mean_reward, step=i)

                    if mean_reward > reward_target:
                        print("Problem Solved")
                        break

            if i % update_steps == 0:
                print("Target Model Updating")
                agent.tg_net.set_weights(agent.net.get_weights())

    except KeyboardInterrupt:
        pass
    finally:
        demo_env = gym.make("LunarLander-v3", render_mode="rgb_array")
        obs, _ = demo_env.reset()
        frames = []
        while True:
            frame = demo_env.render()
            frames.append(frame)
            action = tf.argmax(model(obs[np.newaxis]), axis=-1).numpy()[0]
            obs, reward, terminated, truncated, _ = demo_env.step(action)
            if terminated or truncated:
                break

        import imageio
        imageio.mimsave("DQN_LunarLander.gif", frames, fps=30)

        model.save("dqn.keras")
        tg_model.save("target_dqn.keras")
        env.close()
        test_env.close()
        demo_env.close()
        test_summary_writer.close()
        train_summary_writer.close()


if __name__ == "__main__":
    main()
