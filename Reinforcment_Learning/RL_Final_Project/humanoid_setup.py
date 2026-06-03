import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import Model

class PolicyNetwork(Model):
    def __init__(self, obs_space_dim: int, action_space_dim: int):
        super().__init__()
        
        input_layer = keras.layers.Input((obs_space_dim,))
        hidden_layer1 = keras.layers.Dense(obs_space_dim * 3, activation="elu")(input_layer)
        hidden_layer2 = keras.layers.Dense(obs_space_dim * 2, activation="elu")(hidden_layer1)
        
        self.mean_output_layer = keras.layers.Dense(action_space_dim, activation="tanh")(hidden_layer2)
        self.std_output_layer = keras.layers.Dense(action_space_dim, activation="softmax")(hidden_layer2)
        
        self.model = Model(inputs=[input_layer], outputs=[self.mean_output_layer, self.std_output_layer])
        
    def call(self, state):
        return self.model(state)
    
class REINFORCE:
    def __init__(self, obs_space_dim: int, action_space_dim: int):
        
        self.learing_rate = .05
        self.gamma = .99
        self.policy = PolicyNetwork(obs_space_dim, action_space_dim)
        self.optimizer = keras.optimizer.Nadam(self.learing_rate)
        self.tape = tf.GradientTape(persistent=True)
        
        self.all_rewards = []
        self.all_log_prob = []
    
    def discount_rewards(self, rewards):
        for i in range(len(rewards) - 2, -1, -1):
            rewards[i] = self.gamma * rewards[i+1]
        return rewards
    
    def normalize_rewards(self, all_rewards):
        discoutned_rewards = [self.discount_rewards(rewards) for rewards in all_rewards]
        flat = np.concatenate(discoutned_rewards)
        mean, std = flat.mean(), flat.std()
        return (discoutned_rewards - mean) / (std + 1e-8)
    
    def play_step(self, state):
        mean, std = self.policy(state)
        distrib = tfp.distributions.Normal(mean, std)
        actions = distrib.sample()
        log_prob = tf.reduce_sum(distrib.log_prob(actions), axis=-1)
        self.all_log_prob.append(log_prob)
        return actions
    
    def update(self):
        discounted_rewards = self.discount_rewards(self.all_rewards)
        discounted_rewards_t = tf.constant(discounted_rewards, dtype=tf.float32)
        
        log_probs_t = tf.stack(self.all_log_prob)
        loss = -tf.reduce_mean(discounted_rewards_t * log_probs_t)
        grads = self.tape.gradient(loss, self.policy.trainable_variables)
        self.optimizer.apply(zip(grads, self.policy.trainable_variables))
        self.tape.__exit__()
        del self.tape
        self.all_rewards = []
        self.all_log_prob = []
        
