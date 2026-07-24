import gymnasium as gym
import numpy as np
import tensorflow as tf

from tensorflow import keras

import argparse
from datetime import datetime

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--update", type=int, default=10)
arg_parser.add_argument("--alpha", type=float, default=0.001)
arg_parser.add_argument("--gamma", type=float, default=0.99)

@tf.function
def predict(model, obs):
    return model(obs)

def compute_gradient(model, obs, loss_fn):
    with tf.GradientTape() as tape:
        proba_distributions = predict(model,  obs)
        action = np.random.choice([0, 1], p=proba_distributions.numpy()[0])
        y_target = tf.one_hot([action], depth=2)
        loss = loss_fn(y_target, proba_distributions)
    return action, tape.gradient(loss, model.trainable_variables)

def play_one_step(env, obs, model, loss_fn):
    obs = tf.constant(obs[np.newaxis])
    action, gradient = compute_gradient(model, obs, loss_fn)
    obs, reward, terminated, truncated, _ = env.step(action)
    return obs, reward, terminated, truncated, gradient

def play_episode(env, model, loss_fn):
    obs, _ = env.reset()
    rewards = []
    gradients = []
    while True:
        obs, reward, terimated, truncated, gradient = play_one_step(env, obs, model, loss_fn)
        rewards.append(reward)
        gradients.append(gradient)
        if terimated or truncated:
            break
    return rewards, gradients

def play_multiple_episodes(env, model, loss_fn, iterations):
    all_rewards, all_gradients = [], []
    for _ in range(iterations):
        rewards, gradients = play_episode(env, model, loss_fn)
        all_rewards.append(rewards)
        all_gradients.append(gradients)
    return all_rewards, all_gradients

def discount_reward(episode_reward, gamma):
    episode_reward = np.array(episode_reward)
    for i in range(len(episode_reward) - 2, -1, -1):
        episode_reward[i] =  episode_reward[i] + gamma * episode_reward[i+1]
    return episode_reward

def discounte_normalize(episodes_reward, gamma):
    discounted_rewards = [discount_reward(episode_reward, gamma) for episode_reward in episodes_reward]
    flat = np.concatenate(discounted_rewards)
    mean = flat.mean()
    std = flat.std()
    return [(discounted_reward - mean) / (std + 1e-8) for discounted_reward in discounted_rewards]

def test(env, model):
    obs, _ = env.reset()
    total_rewards = 0
    while True:
        proba_distribution = model(obs[np.newaxis]).numpy()[0]
        action = np.argmax(proba_distribution)
        obs, reward, termianted, truncated, _ = env.step(action)
        total_rewards += reward
        if termianted or truncated:
            break
    return total_rewards


def main():
    args = arg_parser.parse_args()
    EPISODE_UPDATE = args.update
    ALPHA = args.alpha
    GAMMA = args.gamma

    env = gym.make("CartPole-v1")
    test_env = gym.make("CartPole-v1")

    model = keras.Sequential([
        keras.layers.InputLayer((4,)),
        keras.layers.Dense(128, "relu"),
        keras.layers.Dense(2, activation="softmax")
    ])

    optimizer = keras.optimizers.Nadam(learning_rate=ALPHA)
    loss_fn = keras.losses.CategoricalCrossentropy()

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    test_logs_dir = "logs/dqn/CartPole/test/" + current_time

    test_summary_writer = tf.summary.create_file_writer(test_logs_dir)

    best_reward = None
    iteration = 0
    try:

        while True:
            all_rewards, all_gradients = play_multiple_episodes(env, model, loss_fn, EPISODE_UPDATE)
            all_discounted_normalize_rewards = discounte_normalize(all_rewards, GAMMA)

            all_mean_grads = []
            for var_index in range(len(model.trainable_variables)):
                mean_grads = tf.reduce_mean(
                    [final_reward * all_gradients[episode_index][step][var_index]
                        for episode_index, final_rewards in enumerate(all_discounted_normalize_rewards)
                            for step, final_reward in enumerate(final_rewards)], axis=0)
                all_mean_grads.append(mean_grads)

            optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))

            total_reward = 0
            for _ in range(4):
                reward = test(test_env, model)
                total_reward += reward
                if best_reward is None or best_reward < reward:
                    print(f"Update best reward {best_reward} -> {reward}")
                    best_reward = reward
            mean_reward = total_reward / 4

            with test_summary_writer.as_default():
                tf.summary.scalar("test_mean_reward", mean_reward, step=iteration)

            iteration += 1

            if mean_reward > 490:
                model.save("./reinforceMethod_CartPole.keras")
                print("Problem Solved")
                break
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        test_env.close()
    
    demo = gym.make("CartPole-v1", render_mode="rgb_array")
    obs, _ = demo.reset()
    frames = []
    for _ in range(500):
        frame = demo.render()  # returns RGB array
        frames.append(frame)
        action = np.argmax(model(obs[np.newaxis]).numpy()[0])
        obs, _, done, truncated, _ = demo.step(action)
        if done or truncated:
            break
    # Save as GIF using imageio
    import imageio
    imageio.mimsave("cartpole_demo.gif", frames, fps=30)
    print("Saved demo to cartpole_demo.gif")

if __name__ == "__main__":
    main()