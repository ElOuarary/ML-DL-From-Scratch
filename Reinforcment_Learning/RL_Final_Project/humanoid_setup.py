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
        
        self.all_rewards = []
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
    
    def play_step(self, state):
        mean, std = self.policy(state[np.newaxis])
        distrib = tfp.distributions.Normal(mean, std)
        actions = distrib.sample()
        log_prob = tf.reduce_sum(distrib.log_prob(actions), axis=-1)
        self.all_log_prob.append(log_prob)
        return actions[0].numpy()
    
    def update(self, tape):
        discounted_rewards = self.discount_rewards(self.all_rewards)
        discounted_rewards_t = tf.constant(discounted_rewards, dtype=tf.float32)
        
        log_probs_t = tf.stack(self.all_log_prob)
        log_probs_t = tf.where(tf.math.is_nan(log_probs_t), 0, log_probs_t)
        print(f"Trainable variables: {len(self.policy.trainable_variables)}")
        loss = -tf.reduce_mean(discounted_rewards_t * log_probs_t)
        print(f"Loss: {loss}")
        grads = tape.gradient(loss, self.policy.trainable_variables)
        print(grads)
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))
        
        self.all_rewards = []
        self.all_log_prob = []

    def store_rewards(self, reward):
        self.all_rewards.append(reward)
        
def train():
    env = gym.make("HumanoidStandup-v5", render_mode=None)
    agent = REINFORCE(348, 17)
    for i in range(1, 2):
        for episode in range(1):
            print(f"Iteration {i} _ Episode {episode}")
            rewards = 0
            obs, _ = env.reset()
            with tf.GradientTape() as tape:
                while True:
                    actions = agent.play_step(obs)
                    obs, reward, terminated, truncated, _ = env.step(actions)
                    rewards += reward
                    agent.store_rewards(reward)
                    if terminated or truncated: break
            print(f"Total Rewards: {rewards}") 
            agent.update(tape)
            
    env.close()
    
if __name__ == "__main__":
    train()