import gymnasium as gym
import numpy as np
import tensorflow as tf

from tensorflow import keras

class Agent:
    def __init__(self, env, net, loss_fn, optimizer, gamma, episodes):
        self.env = env
        self.actions = range(2)
        self.net = net
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.gamma = gamma
        self.max_episodes = episodes

    def play_step(self, state):
        with tf.GradientTape() as tape:
            proba_distribution = self.net(state)
            action = np.random.choice(self.actions, p=proba_distribution.numpy()[0])
            y_target = tf.reshape(tf.one_hot(self.actions, 1), (1, 2))
            loss = self.loss_fn(y_target, proba_distribution)
        gradient = tape.gradient(loss, self.net.trainable_variables)
        return action, gradient

    def play_epsisode(self):
        state, _ = self.env.reset()
        rewards = []
        gradients = []
        while True:
            action, gradient = self.play_step(state[np.newaxis])
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            state = next_state
            rewards.append(reward)
            gradients.append(gradient)
            if terminated or truncated:
                break
        return rewards, gradients
    
    def discount_reward(self, rewards):
        rewards = np.array(rewards)
        for i in range(-1, -len(rewards), -1):
            rewards[i] = rewards[i] + self.gamma * rewards[i+1]
        return rewards
    
    def baseline_rewards(self, episodes_rewards):
        discounted_rewards = [self.discount_reward(episode_reward) for episode_reward in episodes_rewards]
        flat_rewards = np.concatenate(discounted_rewards)
        mean = flat_rewards.mean()
        std = flat_rewards.std()
        return [(discounted_reward - mean) / (std + 1e-8) for discounted_reward in discounted_rewards]
    
    def train(self):
        all_rewards, all_gradients = [], []
        for _ in range(self.max_episodes):
            episode_reward, episode_gradient = self.play_epsisode()
            all_rewards.append(episode_reward)
            all_gradients.append(episode_gradient)
        all_discounted_rewards = self.baseline_rewards(all_rewards)
        scaled_gradient = []
        for i in range(len(self.net.trainable_variables)):
            mean_grads = tf.reduce_mean([
                reward * all_gradients[episode_idx][step_idx][i]
                for episode_idx, episode in enumerate(all_discounted_rewards)
                    for step_idx, reward in enumerate(episode)
            ], axis=0)
            scaled_gradient.append(mean_grads)
        self.optimizer.apply_gradients(zip(scaled_gradient, self.net.trainable_variables))

    def test(self, test_env):
        state, _ = test_env.reset()
        total_reward = 0
        while True:
            proba = self.net(state[np.newaxis]).numpy()[0]
            action = np.random.choice(self.actions, p=proba)
            state, reward, terminated, truncated, _ = test_env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return total_reward

def main():
    env = gym.make("CartPole-v1")
    test_env = gym.make("CartPole-v1")
    model = keras.Sequential([
        keras.layers.InputLayer(shape=(4,)),
        keras.layers.Dense(128),
        keras.layers.Dense(2,activation="softmax")
    ])
    loss_fn = keras.losses.BinaryCrossentropy()
    optimizer = keras.optimizers.Adam()
    agent = Agent(env, model, loss_fn, optimizer, 0.95, 4)

    iteration = 1
    best_reward = None
    while True:
        total_rewards = 0
        agent.train()

        for _ in range(4):
            episode_reward = agent.test(test_env)
            if best_reward is None or best_reward < episode_reward:
                best_reward = episode_reward
                print(f"Best reward update {best_reward} -> {episode_reward}")

            total_rewards += episode_reward
            
        mean_rewards = total_rewards / 4
        print(f"Iteration {iteration} - Mean Reward: {mean_rewards}")

        if mean_rewards > 195:
            print("Problem Solved")
            break
        iteration += 1

    demo_env = gym.make("CartPole-v1", render_mode="human")
    agent.test(demo_env)
        


if __name__ == "__main__":
    main()