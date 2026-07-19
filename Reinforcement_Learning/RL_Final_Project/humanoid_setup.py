import gymnasium as gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import Model

class PolicyNetwork(Model):
    def __init__(self, obs_space_dim: int, action_space_dim: int):
        super().__init__()
        
        self.hidden_layer1 = keras.layers.Dense(obs_space_dim * 3, activation="elu")
        self.hidden_layer2 = keras.layers.Dense(obs_space_dim * 2, activation="elu")
        self.mean_output_layer = keras.layers.Dense(action_space_dim, activation="tanh")
        self.std_output_layer = keras.layers.Dense(action_space_dim, activation="softplus")
        
        
    def call(self, state):
        x = self.hidden_layer1(state)
        x = self.hidden_layer2(x)
        return self.mean_output_layer(x), self.std_output_layer(x)
    
class REINFORCE:
    def __init__(self, obs_space_dim: int, action_space_dim: int):
        self.learning_rate = 5e-3
        self.gamma = .99
        self.policy = PolicyNetwork(obs_space_dim, action_space_dim)
        self.optimizer = keras.optimizers.Nadam(self.learning_rate)
        self.means = []
        self.variance = []
        self.rewards = []
        self.actions = []
        self.all_log_prob = []
    
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        for i in range(len(rewards) - 2, -1, -1):
            discounted_rewards[i] = rewards[i] + self.gamma * rewards[i+1]
        return discounted_rewards
    
    def normalize_rewards(self, all_rewards):
        discoutned_rewards = [self.discount_rewards(rewards) for rewards in all_rewards]
        flat = np.concatenate(discoutned_rewards)
        mean, std = flat.mean(), flat.std()
        return (discoutned_rewards - mean) / (std + 1e-8)
    
    def play_step(self, env, state):
        mean, var = self.policy(state[np.newaxis])
        distrib = tfp.distributions.Normal(mean, var)
        actions = distrib.sample()
        next_state, reward, done, truncated, _ = env.step(actions[0])
        self.means.append(mean)
        self.variance.append(var)
        self.rewards.append(reward)
        self.actions.append(actions)
        return next_state, reward, done, truncated
    
    def update(self):
        self.rewards = self.discount_rewards(self.rewards)
        self.means = np.array(self.means)
        self.variance = np.array(self.variance)
        self.rewards = np.array(self.rewards)
        self.actions = np.array(self.actions)
        with tf.GradientTape() as tape:
            p1 = - (self.means - self.actions) ** 2 / (2 * self.variance + 1e-8)
            p2 = tf.math.log(tf.math.sqrt(2 * np.pi * self.variance))
            log_loss = tf.reduce_mean(p1 + p2, axis=1)
            loss = - tf.reduce_sum(log_loss * self.rewards)
        grads = tape.gradient(loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))
        self.rewards = []
        self.means = []
        self.rewards = []
        self.actions = []
        
        
def train():
    env = gym.make("HumanoidStandup-v5", render_mode="human")
    agent = REINFORCE(348, 17)
    for i in range(1, 2):
        for episode in range(1):
            state, _ = env.reset()
            while True:
                state, reward, done, truncated = agent.play_step(env, state)
                if done or truncated:
                    break
            agent.update()
            
    env.close()
    
if __name__ == "__main__":
    train()