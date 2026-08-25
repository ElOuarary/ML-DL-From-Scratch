import argparse
from collections import namedtuple
from datetime import UTC, datetime

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

Step = namedtuple(
    "Step", field_names=("state", "action", "reward", "next_state", "continue_mask")
)

class NoiseDense(keras.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(NoiseDense, self).__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.mu_w = self.add_weight(
            shape=(input_dim, self.units),
            initializer=keras.initializers.RandomUniform(-1/np.sqrt(input_dim), 1/np.sqrt(input_dim)),
            name="mu_w",
        )
        self.sigma_w = self.add_weight(
            shape=(input_dim, self.units),
            initializer=keras.initializers.Constant(0.017),
            name="sigma_w"
        )

        self.mu_b = self.add_weight(
            shape=(self.units,),
            initializer=keras.initializers.RandomUniform(-1/np.sqrt(input_dim), 1/np.sqrt(input_dim)),
            name="mu_b"
        )
        self.sigma_b = self.add_weight(
            shape=(self.units,),
            initializer=keras.initializers.Constant(0.017),
            name="sigma_b"
        )
        
    def call(self, inputs):
        epsilon_w = tf.random.normal(shape=(tf.shape(inputs)[-1], self.units))
        epsilon_b = tf.random.normal(shape=(self.units,))
        weights = self.mu_w + self.sigma_w * epsilon_w
        bias = self.mu_b + self.sigma_b * epsilon_b

        output = tf.matmul(inputs, weights) + bias
        if self.activation is not None:
            output = self.activation(output)

        return output

class PrioritezReplayBuffer:
    def __init__(self, buff_size, proba_alpha, beta_start, beta_frames):
        self.proba_alpha = proba_alpha
        self.capacity = buff_size
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros((buff_size, ), dtype=np.float32)
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_frames = beta_frames

    def update_beta(self, idx):
        v = self.beta_start + idx (1 - self.beta_start) / self.beta_frames
        self.beta = min(1, v)

    def __len__(self):
        return len(self.buffer)

    def populate(self, sample):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.pos] = sample
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probas = prios**self.proba_alpha
        probas /= np.sum(probas)
        indices = np.random.choice(len(self.buffer), batch_size ,p=probas)
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probas[indices]) ** (-self.beta)
        weights /= weights.max()
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

class Agent:
    def __init__(
        self, env, gamma, net, tg_net, optimizer, n_steps, buffer
    ):
        self.env = env
        self.state, _ = self.env.reset()
        self.action_space = env.action_space.n
        self.gamma = gamma
        self.net = net
        self.tg_net = tg_net
        self.optimizer = optimizer
        self.n_steps = n_steps
        self.replay_buffer = buffer

    def explore(self):
        action = self.greedy_policy(self.state)
        action = action.numpy()[0]
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        self.replay_buffer.populate(
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

    # Problem with prioritized replay buffer due to it circular nature
    def compute_returns(self, indices, rewards, continues):
        returns = []
        next_state = []
        continue_mask = []
        for i, reward, continue_ in zip(indices, rewards, continues):
            j = i
            if continue_:
                for j in range(i+1, min(i+self.n_steps, len(self.replay_buffer))):
                    reward += self.gamma ** (j-i) * self.replay_buffer.buffer[j].reward
                    if not self.replay_buffer.buffer[j].continue_mask:
                        break
                next_state.append(self.replay_buffer.buffer[j].next_state)
                continue_mask.append(self.replay_buffer.buffer[j].continue_mask if i+self.n_steps <= len(self.replay_buffer) else 0)
            else:
                next_state.append(self.replay_buffer.buffer[i].next_state)
                continue_mask.append(self.replay_buffer.buffer[i].continue_mask)
            returns.append(reward)
        return returns, next_state, continue_mask

    def sample_batch(self, batch_size):
        samples, indices, weights = self.replay_buffer.sample(batch_size)
        states, actions, rewards, _, continue_mask = zip(
            *[sample for sample in samples]
        )
        
        retruns, next_states, continue_mask = self.compute_returns(indices, rewards, continue_mask)
        states, actions, retruns, next_states, continue_mask, weights = (
            tf.constant(states),
            tf.constant(actions),
            tf.constant(retruns, dtype=tf.float32),
            tf.constant(next_states),
            tf.constant(continue_mask, dtype=tf.float32),
            tf.constant(weights)
        )
        return states, actions, retruns, next_states, continue_mask, weights, indices

    @tf.function
    def greedy_policy(self, state):
        return tf.argmax(self.net(state[np.newaxis]), axis=-1)

    @tf.function
    def compute_loss(self, state, reward, action, next_state, continue_mask, batch_weights):
        next_state_optimal_action = tf.argmax(self.net(next_state), axis=-1)
        next_state_value = tf.gather(self.tg_net(next_state), next_state_optimal_action, batch_dims=1)

        q_value_target = reward + self.gamma ** self.n_steps * continue_mask * next_state_value
        mask = tf.one_hot(action, self.action_space)
        with tf.GradientTape() as tape:
            q_value = self.net(state)
            q_value_masked = tf.reduce_sum(mask * q_value, axis=-1)
            loss = (q_value_target - q_value_masked) ** 2
            weighted_loss = batch_weights * loss
            mean_loss = tf.reduce_mean(weighted_loss)
        gradients = tape.gradient(mean_loss, self.net.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.net.trainable_variables))
        return mean_loss, gradients, weighted_loss + 1e-5

    @tf.function
    def compute_batch(self, states):
        return self.net(states)

    def train_model(self, batch_size):
        states, actions, rewards, next_states, continue_mask, weights, indices = self.sample_batch(
            batch_size
        )
        loss, gradient, batch_loss = self.compute_loss(states, rewards, actions, next_states, continue_mask, weights)
        self.replay_buffer.update_priorities(indices, batch_loss)
        return loss, gradient

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
    parser.add_argument("--n-steps", type=int, default=4)
    parser.add_argument("--proba-alpha", type=float, default=0.6)
    parser.add_argument("--beta-start", type=float, default=0.4)
    parser.add_argument("--beta-steps", type=int, default=10_000)

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
    n_steps = args.n_steps
    proba_alpha = args.proba_alpha
    beta_start = args.beta_start
    beta_steps = args.beta_steps

    # How to link the experience gathering with this object
    buffer = PrioritezReplayBuffer(buffer_size, proba_alpha, beta_start, beta_steps)

    env = gym.make("LunarLander-v3")
    test_env = gym.make_vec("LunarLander-v3", num_envs=20)
    model = keras.Sequential(
        [
            keras.layers.InputLayer(env.observation_space.shape),
            NoiseDense(256, activation="relu"),
            NoiseDense(256, activation="relu"),
            keras.layers.Dense(4),
        ]
    )
    tg_model = keras.models.clone_model(model)
    tg_model.set_weights(model.get_weights())

    optimizer = keras.optimizers.Nadam(learning_rate=alpha, clipnorm=1)
    agent = Agent(env, gamma, model, tg_model, optimizer, n_steps, buffer)

    current_time = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    train_logs_dir = "logs/dqn/LunarLander/train/" + current_time
    test_logs_dir = "logs/dqn/LunarLander/test/" + current_time

    train_summary_writer = tf.summary.create_file_writer(train_logs_dir)
    test_summary_writer = tf.summary.create_file_writer(test_logs_dir)

    try:
        for i in range(1, train_iteration + 1):
            agent.explore()

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
            obs, _, terminated, truncated, _ = demo_env.step(action)
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
