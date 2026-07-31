import gymnasium as gym
import ale_py
import tensorflow as tf
from tensorflow import keras

gym.register_envs(ale_py)

class Actor2Critic(keras.Model):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.shared_network = keras.models.Sequential([
            keras.layers.Conv2D(32, kernel_size=(3, 3),activation="relu", input_shape=observation_space),
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dense(256, activation="relu")
        ])
        self.actor = keras.layers.Dense(18, activation="softmax")
        self.critic = keras.layers.Dense(1)

    @tf.function
    def call(self, obs):
        x = self.shared_network(obs)
        return self.actor(x), self.critic(x)

def main():
    try:
        env = gym.make("ALE/BattleZone-v5")
        model = Actor2Critic(env.observation_space.shape, env.action_space.n)
        obs, _ = env.reset()
        total_reward = 0
        while True:
            obs = tf.constant([obs])
            proba_disctribution, _ = model(obs)
            action = tf.argmax(proba_disctribution, axis=-1).numpy()[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                print(f"Total Reward: {total_reward}")
                total_reward = 0
                obs, _ = env.reset()

    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()