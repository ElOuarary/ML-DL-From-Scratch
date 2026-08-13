import tensorflow as tf
import numpy as np
from gymnasium.vector import SyncVectorEnv
from a2c import Actor2Critic
from utils import make_atari_env

envs = SyncVectorEnv([make_atari_env("ALE/BattleZone-v5") for _ in range(8)])
envs.num_envs
model = Actor2Critic((84, 84, 4), 18)

state, _ = envs.reset()

states, actions, rewards, dones = [], [], [], []

for _ in range(5):
    state = np.transpose(state.astype(np.float32), axes=(0, 2, 3, 1)) / 255.0
    action_logits, state_values = model(state)
    action = tf.random.categorical(action_logits, num_samples=1)[:, 0].numpy()
    next_state, reward, terminated, truncated, _ = envs.step(action)

    states.append(state)
    actions.append(action)
    rewards.append(reward)
    dones.append(terminated | truncated)
    state = next_state

states = np.stack(states, axis=0)
actions = np.stack(actions, axis=0)
rewards = np.stack(rewards, axis=0)
dones = np.stack(dones, axis=0)


next_state = np.transpose(next_state.astype(np.float32), axes=(0, 2, 3, 1)) / 255.0
_, next_state_value = model(next_state)
boostrapped_values = next_state_value.numpy().flatten()

boostrapped_values = np.array([10.0, 0.0])

returns = np.zeros_like(rewards, dtype=np.float32)
for env_id in range(2):
    R = boostrapped_values[env_id]
    for i in reversed(range(3)):
        if dones[i, env_id]:
            R = rewards[i, env_id]
        else:
            R = rewards[i, env_id] + 0.99 * R
        returns[i, env_id] = R

print(returns)