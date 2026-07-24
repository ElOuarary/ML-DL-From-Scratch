import argparse
from collections import namedtuple
from datetime import datetime

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--batch_size", default=32, type=int)
arg_parser.add_argument("--percentile", default=70, type=int)
arg_parser.add_argument("--max_iterations", default=1000, type=int)
arg_parser.add_argument("--target_rewards", default=-60, type=int)
arg_parser.add_argument("--render", default=False, type=bool)

Episode = namedtuple("Episode", ["steps", "total_reward"])
EpisodeStep = namedtuple("EpisodeStep", ["observation", "action"])

@tf.function
def infernece(model, obs_batch):
    return model(obs_batch)

def generate_batch(env, model, batch_size):
    batch = []
    steps = []
    episode_reward = 0
    obs, _ = env.reset()
    obs = tf.reshape(tf.constant(obs), (1, -1))
    while True:
            action_probs = infernece(model, obs).numpy()[0]
            probs = action_probs / sum(action_probs)
            action = np.random.choice(range(env.action_space.n), p=probs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            step = EpisodeStep(obs, action)
            steps.append(step)
            obs = next_obs
            obs = tf.reshape(tf.constant(obs), (1, -1))
            if terminated or truncated:
                batch.append(Episode(steps, episode_reward))
                episode_reward = 0
                steps = []
                obs, _ = env.reset()
                obs = tf.reshape(tf.constant(obs), (1, -1))
            if len(batch) == batch_size:
                yield batch
                batch = []

def filter_episode(batch, percentile):
    rewards = list(map(lambda x: x.total_reward, batch))
    rewards_mean = np.mean(rewards)
    rewards_bounds = np.percentile(rewards, percentile)

    train_obs = []
    train_act = []
    elite_episode = []
    for episode in batch:
        if episode.total_reward > rewards_bounds:
            train_obs.extend(map(lambda x: x.observation, episode.steps))
            train_act.extend(map(lambda x: x.action, episode.steps))
    return train_obs, train_act, rewards_mean, rewards_bounds, elite_episode

def compute_apply_gradient(model, obs_v, y_target, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        y_pred = infernece(model, obs_v)
        loss = loss_fn(y_target, y_pred)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss
    
def main():
    args = arg_parser.parse_args()
    BATCH_SIZE = args.batch_size
    PERCENTILE = args.percentile
    MAX_ITERATIONS = args.max_iterations
    TARGET_REWARDS = args.target_rewards
    elite_episode = []

    if args.render:
        env = gym.make("Acrobot-v1", render_mode="human")
    else:
        env = gym.make("Acrobot-v1")

    model = keras.Sequential([
    keras.layers.InputLayer(shape=(6,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(3, activation="softmax")
    ])

    optimizer = keras.optimizers.Nadam(learning_rate=0.005)
    loss_fn = keras.losses.CategoricalCrossentropy()
    
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = "logs/crossEntropyMethod/acrobat/" + current_time
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    
    try:   
        for i, episodes in enumerate(generate_batch(env, model, BATCH_SIZE)):
            obs_v, act_v, rewards_mean, rewards_boundary, elite_episode = filter_episode(elite_episode + episodes, PERCENTILE)
            if not obs_v:
                continue
            y_target = tf.one_hot(act_v, 3)
            obs_v = tf.reshape(tf.Variable(obs_v), (-1, 6))
            loss = compute_apply_gradient(model, obs_v, y_target, loss_fn, optimizer)

            elite_episode.sort()
            elite_episode = elite_episode[-10:]

            with train_summary_writer.as_default():
                tf.summary.scalar("rewards_mean", rewards_mean, step=i)
                tf.summary.scalar("rewards_boudary", rewards_boundary, step=i)
                tf.summary.scalar("loss", loss, step=i)

            if i >= MAX_ITERATIONS or rewards_mean >= TARGET_REWARDS:
                break

        demo = gym.make("Acrobot-v1", render_mode="rgb_array")
        obs, _ = demo.reset()
        frames = []
        for _ in range(500):
            frame = demo.render()
            frames.append(frame)
            action = np.argmax(model(obs[np.newaxis]).numpy()[0])
            obs, _, done, truncated, _ = demo.step(action)
            if done or truncated:
                break
        # Save as GIF using imageio
        import imageio
        imageio.mimsave("cartpole_demo.gif", frames, fps=30)
        print("Saved demo to cartpole_demo.gif")
    except KeyboardInterrupt as e:
        tf.print("Stoping experiment")
    else:
        env.close()
        demo.close()
        model.save("./crossEntrorpyMethod.keras")

if __name__ == "__main__":
    main()