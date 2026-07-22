import gymnasium as gym
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, losses, optimizers, Model


def main():
    env = gym.make("LunarLander-v3")
    test_env = gym.make("LunarLander-v3")

    inputs = layers.Input(shape=(8,))
    common = layers.Dense(128, activation="relu")(inputs)
    action = layers.Dense(4, activation="softmax")(common)
    critic = layers.Dense(1)(common)

    model = Model(inputs=inputs, outputs=[action, critic])

    optimizer = optimizers.Nadam(learning_rate=0.001)
    loss_fn = losses.MeanSquaredError()

    action_proba_history = []
    value_state_history = []
    reward_history = []

    episode = 0
    best_reward = None

    while True:
        obs, _ = env.reset()
        episode_reward = 0
        with tf.GradientTape() as tape:
            while True:
                actions_probs, value = model(obs[np.newaxis])
                action = np.random.choice(range(4), p=actions_probs.numpy()[0])
                value_state_history.append(value[0, 0])
                action_proba_history.append(tf.math.log(actions_probs[0, action]))

                obs, reward, terminated, truncated, _ = env.step(action)
                reward_history.append(reward)

                episode_reward += reward
                if terminated or truncated:
                    break
            
            action_returns = []
            returns = 0
            for i in range(len(reward_history) - 1, -1, -1):
                returns = reward_history[i] + 0.95 * returns
                action_returns.insert(0, returns)

            history = zip(action_proba_history, value_state_history, action_returns)
            actor_loss = []
            critic_loss = []

            for log_action, value, ret in history:
                actor_loss.append(-(ret - value) * log_action)
                critic_loss.append(loss_fn(tf.expand_dims(ret, 0), tf.expand_dims(value, 0)))

            loss_value = sum(actor_loss) + sum(critic_loss)
        
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        action_proba_history.clear()
        value_state_history.clear()
        reward_history.clear()

        test_reward = 0
        obs, _ = test_env.reset()
        while True:
            action_proba, _ = model(obs[np.newaxis])
            action = np.argmax(action_proba)
            obs, reward, truncated, terminated, _ = test_env.step(action)
            test_reward += reward

            if terminated or truncated:
                break

        print(f"Episode {episode} - Test Reward: {test_reward}")
        if best_reward is None or best_reward < test_reward:
            print(f"Best Reward Update {best_reward} -> {test_reward}")
            best_reward = test_reward
        episode += 1

        if test_reward > 200:
            print("Solved!")


if __name__ == "__main__":
    main()