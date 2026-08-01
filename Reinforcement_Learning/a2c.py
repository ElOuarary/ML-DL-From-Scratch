import gymnasium as gym
import ale_py
import numpy as np
import tensorflow as tf
from tensorflow import keras

gym.register_envs(ale_py)

env = gym.make("ALE/BattleZone-v5")

class Actor2Critic(keras.Model):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.shared_network = keras.models.Sequential([
            keras.layers.InputLayer(observation_space),
            keras.layers.Conv2D(32, kernel_size=8, strides=4,activation="relu"),
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

@tf.numpy_function(Tout=[tf.float32, tf.int32, tf.int32])
def env_step(action):
    next_obs, reward, termiated, truncated, _ = env.step(action)
    return next_obs, reward, termiated | truncated

@tf.function
def play_episode(initial_state, model):
    obs = tf.cast(initial_state, tf.float32)
    actions_probas = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    i = 0
    while tf.constant(True):
        action_logits, state_value = model(obs[np.newaxis])
        action = tf.random.categorical(action_logits, 1)[0, 0]
        action_proba = tf.nn.softmax(action_logits)
        obs, reward, done= env_step(action)

        actions_probas = actions_probas.write(i, action_proba[0, action])
        values = values.write(i, tf.squeeze(state_value))
        rewards = rewards.write(i, reward)
        if tf.cast(done, tf.bool):
            break
        i += 1

    return actions_probas.stack(), values.stack(), rewards.stack()

def discount_reward(rewards, gamma):
    discounted_reward = np.array(rewards)
    for i in range(len(rewards) - 2, -1, -1):
        discounted_reward[i] += gamma * discounted_reward[i+1]
    return (discounted_reward - discounted_reward.mean()) / (discounted_reward.std() - 1e-8)

def play_test_episode(test_env, model):
    states, _ = test_env.reset()
    while True:
        action_logits, _ = model(states)
        action_probas = tf.nn.softmax(action_logits)
        optimal_action = tf.argmax(action_probas, axis=-1).numpy()[0]
        states, reward, terminated, truncated, _ = test_env.step(optimal_action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward

def main():
    env = gym.make("ALE/BattleZone-v5")
    test_env = gym.make("ALE/BattleZone-v5")
    model = Actor2Critic(env.observation_space.shape, 18)
    gamma = 0.99
    reward_threhold = 3000

    critic_loss_fn = keras.losses.Huber()
    optimizer = keras.optimizers.Nadam()

    iteration = 1

    while True:
        with tf.GradientTape() as tape:
            initial_state, _ = env.reset()
            actions_probas, values, rewards = play_episode(initial_state, model)
            actor_loss = - tf.reduce_mean((rewards - values) * tf.math.log(actions_probas))
            discounted_rewards = discount_reward(rewards, gamma)
            critic_loss = critic_loss_fn(discounted_rewards, values)
            loss = actor_loss + critic_loss

        print("Loss", loss)
        grad = tape.gradient(loss, model.trainable_variables)
        print("Grad", grad)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))


        if iteration % 1000:
            test_reward = play_test_episode(test_env, model)
            print(f"Iteration {iteration} - Average Reward: {test_reward}")
            if test_reward > reward_threhold:
                print("Problem Solved!")

        iteration += 1

if __name__ == "__main__":
    main()