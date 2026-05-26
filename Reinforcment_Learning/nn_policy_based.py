import gymnasium as gym
import multiprocessing as mp
import numpy as np
from tensorflow import keras
import tensorflow as tf

def run_episode(args):
    weights, n_max_steps, obs_mean, obs_std = args
    
    model = build_model()
    model.set_weights(weights)
    loss_fn = keras.losses.CategoricalCrossentropy(from_logits=False)
    
    env = gym.make("Acrobot-v1")
    obs, _ = env.reset()
    
    rewards, gradients = [], []
    
    for _ in range(n_max_steps):
        obs_norm = (obs - obs_mean) / (obs_std + 1e-8)
        obs_t = tf.convert_to_tensor(obs_norm[np.newaxis], dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            y_predict = model(obs_t)
            action = np.random.choice(y_predict.shape[1], p=y_predict.numpy()[0])
            y_target = np.zeros(y_predict.shape, dtype=np.float32)
            y_target[0, action] = 1
            loss = loss_fn(y_target, y_predict)
            
        grads = tape.gradient(loss, model.trainable_variables)
        grads_np = [g.numpy() for g in grads]
        
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        gradients.append(grads_np)
        
        if terminated or truncated:
            break
        
    return rewards, gradients

def build_model():
    return keras.Sequential([
        keras.layers.Dense(6, activation="relu"),
        keras.layers.Dense(64, activation=keras.activations.relu),
        keras.layers.Dense(3, activation="softmax")
    ])

def play_one_step(env, obs, model, loss_fn):
    obs_norm = (obs - OBS_MEAN) / (OBS_STD + 1e-8)
    with tf.GradientTape() as tape:
        y_pred = model(obs_norm[np.newaxis])
        action = np.random.choice(len(y_pred[0]), p=y_pred[0].numpy())
        y_target = np.zeros(y_pred.shape)
        y_target[0, action] = 1
        loss = loss_fn(y_target, y_pred)
    
    grad = tape.gradient(loss, model.trainable_variables)
    obs, reward, terminated, truncated, info = env.step(action)
    return obs, reward, terminated, truncated, grad

def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
    all_rewards = []
    all_gradients = []
    for episode in range(n_episodes):
        obs, info = env.reset()
        rewards = []
        gradients = []
        for step in range(n_max_steps):
            print(f"Episode: {episode} _ Step: {step}")
            obs, reward, terminated, truncated, grad = play_one_step(env, obs, model, loss_fn)
            rewards.append(reward)
            gradients.append(grad)
            if terminated or truncated:
                break      
        all_rewards.append(rewards)
        all_gradients.append(gradients)
    return all_rewards, all_gradients

def discount_rewards(rewards, discount_factor):
    discounted = np.array(rewards, dtype=np.float64)
    for step in range(len(rewards) - 2, -1, -1):
        discounted[step] += discounted[step + 1] * discount_factor
    return discounted

def discount_and_normalize_rewards(all_rewards, discount_factor):
    all_discounted_rewards = [discount_rewards(reward, discount_factor) for reward in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    mean = flat_rewards.mean()
    std = flat_rewards.std()
    return [(discounted_reward - mean) / (std + 1e-8) for discounted_reward in all_discounted_rewards]
    
N_ITERATIONS = 150
N_EPISODES_PER_ITERATION = 30
N_MAX_STEPS = 500
DISCOUNT_FACTOR = 0.99
N_WORKERS = min(mp.cpu_count(), 2)

OBS_MEAN = np.array([0., 0., 0., 0., 0., 0.], dtype=np.float32)
OBS_STD  = np.array([1., 1., 1., 1., 8., 14.], dtype=np.float32)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    
    model = build_model()
    optimizer = keras.optimizers.Nadam(learning_rate=0.005)
    
    with mp.Pool(processes=N_WORKERS) as pool:
        for iteration in range(N_ITERATIONS):
            weights = model.get_weights()
            args = [(weights, N_MAX_STEPS, OBS_MEAN, OBS_STD)] * N_EPISODES_PER_ITERATION
            results = pool.map(run_episode, args)
            all_rewards = [r for r, _ in results]
            all_gradients = [g for _, g in results]
            
            all_norm_rewards = discount_and_normalize_rewards(all_rewards, DISCOUNT_FACTOR)
            
            n_vars = len(model.trainable_variables)
            mean_grads = []
            
            for var_idx in range(n_vars):
                weighted = [
                    step_reward * all_gradients[ep_idx][step][var_idx]
                    for ep_idx, ep_rewards in enumerate(all_norm_rewards)
                        for step, step_reward in enumerate(ep_rewards)
                ]
                mean_grads.append(tf.reduce_mean(weighted, axis=0))

        optimizer.apply_gradients(zip(mean_grads, model.trainable_variables))
        mean_ep_reward = np.mean([sum(r) for r in all_rewards])
        print(f"{iteration:3d} mean episode {mean_ep_reward:.1f}")
    
    model.save("acrobot_policy.keras")