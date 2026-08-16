from gymnasium.vector import AsyncVectorEnv
import numpy as np
from tensorflow.keras.models import load_model
import a2c

from utils import make_atari_env

test_env = AsyncVectorEnv([make_atari_env("ALE/BattleZone-v5") for _ in range(5)])

model = a2c.Actor2Critic((84, 84, 4), 18)

model.build((84, 84, 4))

model.load_weights("a2c.weights.h5")

model.save("a2c.keras")

model2 = load_model("a2c.keras")


# imageio.mimsave("BatlleZone-demo.gif", frames, fps=30)
states, _ = test_env.reset()

total_reward = 0
while True:
    states = np.transpose(states, axes=(0, 2, 3, 1))
    states = states.astype(np.float32) / 255.0
    action_logits, _ = model(states, training=False)
    optimal_action = action_logits.numpy().argmax(axis=-1)
    states, reward, terminated, _, _ = test_env.step(optimal_action)
    total_reward += reward
    print(terminated)
    if np.any(terminated):
        break

print(np.mean(total_reward))