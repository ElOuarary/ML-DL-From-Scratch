import imageio
import numpy as np
from tensorflow.keras.models import load_model
import a2c

from utils import make_atari_env

env = make_atari_env("ALE/BattleZone-v5", render_mode="rgb_array")()

model = a2c.Actor2Critic((84, 84, 4), 18)

model.build((84, 84, 4))

model.load_weights("a2c.weights.h5")

model.save("a2c.keras")

model2 = load_model("a2c.keras")

# frames = []
# obs, _ = env.reset()

# for _ in range(5000):
#     frame = env.render()
#     frames.append(frame)
#     obs = np.transpose(obs, axes=(1, 2, 0)) / 255.0
#     action_logitis, _ = model(obs.astype(np.float32)[np.newaxis])
#     action = action_logitis[0].numpy().argmax(axis=-1)
#     obs, reward, terminated, truncated, _ = env.step(action)
#     if terminated or truncated:
#         break

# imageio.mimsave("BatlleZone-demo.gif", frames, fps=30)