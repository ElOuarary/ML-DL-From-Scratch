import gymnasium as gym
import ale_py
import numpy as np
import tensorflow as tf
from tensorflow import keras

gym.register_envs(ale_py)

class Actor2Critic(keras.Model):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.shared_network = keras.models.Sequential([
            keras.layers.Conv2D(32, kernel_size=8, strides=4,activation="relu", input_shape=observation_space),
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

    @tf.function
    def call(self, obs):
        x = self.shared_network(obs)
        return self.actor(x), self.critic(x)

def env_step(env, obs):
    next_obs, reward, termiated, truncated, _ = env.step(obs)
    return next_obs, reward, termiated | truncated

def play_episode(env, model):
    obs, _ = env.reset()
    actions = []
    actions_logits = []
    values = []
    rewards = []
    while True:
        action_logits, state_value = model(obs[np.newaxis])
        action = tf.random.categorical(actions_logits, 1)[0, 0].numpy()
        obs, reward, terminated, truncated, _ = env.step(action)

        actions.append(action)
        actions_logits.append(action_logits)
        values.append(state_value)
        rewards.append(reward)
        if terminated or truncated:
            break

    return actions, actions_logits, values, rewards

def discount_reward(rewards, gamma):
    discounted_reward = np.array(rewards)
    for i in range(len(rewards) - 2, -1, -1):
        rewards[i] += gamma * rewards[i+1]
    return (discounted_reward - discounted_reward.mean()) / (discounted_reward.std() - 1e-8)

def play_test_episode(test_env, model):
    states, _ = test_env.reset()
    actives = np.ones(test_env.num_envs)
    total_reward = np.zeros(test_env.num_envs)
    while np.any(actives):
        action_logits, _ = model(states)
        action_probas = tf.nn.softmax(action_logits)
        optimal_action = tf.argmax(action_probas, axis=-1).numpy()[0]
        states, reward, terminated, truncated, _ = test_env.step(optimal_action)
        total_reward += reward
        actives = np.logical_and(actives, np.logical_not(termiated | truncated))
    return total_reward.mean()

def main():
    env = gym.make("ALE/BattleZone-v5")
    test_env = gym.make_vec("ALE/BattleZone-v5", 5)
    model = Actor2Critic(env.observation_space.shape, 18)
    gamma = 0.99
    reward_threhold = 3000

    actor_loss_fn = keras.losses.CategoricalCrossentropy()
    critic_loss_fn = keras.losses.Huber()
    optimizer = keras.optimizers.Nadam()

    iteration = 1

    while True:
        with tf.GradientTape() as tape:
            actions, actions_logits, values, rewards = play_episode(env, model)
            actions = tf.one_hot(actions, 18)
            actions_probas = tf.nn.softmax(actions_logits)
            actor_loss = actor_loss_fn(actions, actions_probas)
            discounted_rewards = discount_reward(rewards, gamma)
            critic_loss = critic_loss_fn(discounted_rewards, values)
            loss = actor_loss + critic_loss
        grad = tape.gradient(loss, model.trainable_variables)
        optimizer.apply(zip(grad, model.trainable_variables))

        if iteration % 1000:
            test_reward = play_test_episode(test_env, model)
            print(f"Iteration {iteration} - Average Reward: {test_reward}")
            if test_reward > reward_threhold:
                print("Problem Solved!")

        iteration += 1
        
if __name__ == "__main__":
    main()