import gymnasium as gym
from tensorflow import keras
from tensorflow.keras import Model
import tensorflow as tf
import numpy as np

env = gym.make("HumanoidStandup-v5", render_mode="human")

model = keras.Sequential([
    keras.layers.Dense(348),
    keras.layers.Dense(348 * 3, activation="elu"),
    keras.layers.Dense(348 * 2, activation="elu"),
    keras.layers.Dense(17, activation="softmax"),
])

input_layer = keras.layers.Input((348,))
hidden_layer1 = keras.layers.Dense(348 * 3, activation="elu")(input_layer)
hidden_layer2 = keras.layers.Dense(348 * 2, activation="elu")(hidden_layer1)
policy_mean_layer = keras.layers.Dense(17, activation="tanh")(hidden_layer2)
policy_stddev_layer = keras.layers.Dense(17, activation="tanh")(hidden_layer2)

model = Model(inputs=input_layer, outputs=[policy_mean_layer, policy_stddev_layer])

def play_step(env, obs, loss_fn):
    with tf.GradientTape() as tape:
        y_pred = model.predict(obs[np.newaxis])
        action = np.random.choice(range(len(y_pred[0])), p=y_pred[0])
        y_target = np.zeros(y_pred.shape)
        y_target[0][action] = 1
        loss = loss_fn(y_target, y_pred)
    
    grad = tape.gradient(loss, model.trainable_variables)
    obs, reward, terminated, truncated, info = env.step(action)
    
    return obs, reward, terminated, truncated, info, grad

def discount_reward(rewards, gamma):
    rewards_discounted = np.array(rewards)
    for i in range(len(rewards) - 2, -1, -1):
        rewards_discounted[i] += gamma * rewards_discounted[i+1]
    return rewards_discounted

def normalize_discount_reward(all_rewards, gamma):
    all_rewards_discounted = [discount_reward(rewards, gamma) for rewards in all_rewards]
    flat = np.concatenate(all_rewards_discounted)
    mean, std = flat.mean(), flat.std()
    return [(reward_discounted - mean) / std + 1e-8 for reward_discounted in all_rewards_discounted]
    
def train_model(env, model, iterations, episodes, steps, gamma, loss_fn, optimizer):
    for i in range(iterations):
        all_rewards, all_grads = [], []
        for episode in range(episodes):
            obs, info = env.reset()
            episode_reward, episode_grad = [], []
            for step in range(steps):
                obs, reward, terminated, truncated, info, grad = play_step(env, obs, loss_fn)
                
                episode_reward.append(reward)
                episode_grad.append(grad)
                        
                if terminated or truncated: break
            
            all_rewards.append(episode_reward)
            all_grads.append(episode_grad)
            print(f"Iteration {i} Episode {episode} Total Rewards: {sum(episode_reward)}")
        
        all_rewards_discounted = normalize_discount_reward(all_rewards, gamma)
        
        mean_grads = []
        for weight_idx in range(len(model.trainable_variables)):
            weighted = [
                step_reward * all_grads[ep_idx][step_idx][weight_idx]
                for ep_idx, episode in enumerate(all_rewards_discounted)
                    for step_idx, step_reward in enumerate(episode) 
            ]
            mean_grads.append(tf.reduce_mean(weighted, axis=0))
            
        optimizer.apply(zip(mean_grads, model.trainable_variables))

if __name__ == "__main__":
    print(model(np.random.random(348)[np.newaxis]))