import tensorflow as tf
import numpy as np
from gymnasium.vector import SyncVectorEnv
from a2c import Actor2Critic
from utils import make_atari_env

envs = SyncVectorEnv([make_atari_env("ALE/BattleZone-v5") for _ in range(8)])
envs.num_envs
model = Actor2Critic((84, 84, 4), 18)

state, _ = envs.reset()

train_rewards = tf.constant([[1.,  2. ],
 [0.,  3. ],
 [4.,  0. ]])

train_dones = tf.constant(
    [[False, False],
 [False, True ],
 [False, False]]
)

boostrapped_values = tf.Variable([10.0, 0.0])

returns = tf.Variable(tf.zeros_initializer()(shape=train_rewards.shape, dtype=np.float32))

for i in tf.reverse(tf.range(3), axis=[0]):
    boostrapped_values.assign(tf.where(train_dones[i], train_rewards[i], train_rewards[i] + 0.99 * boostrapped_values))
    returns[i].assign(boostrapped_values)

print(returns)