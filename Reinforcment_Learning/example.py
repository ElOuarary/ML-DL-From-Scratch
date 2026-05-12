from tensorflow import keras
import gymnasium as gym

env = gym.make("Acrobot-v1", render_mode="human")
obs, info = env.reset()

episode_over = False
totals = 0

model = keras.Sequential([
    keras.layers.InputLayer(6),
    keras.layers.Dense(12, activation="relu"),
    keras.layers.Dense(3, activation="softmax")
])

while not episode_over:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    totals += reward
    episode_over = terminated or truncated
    
print(totals)
env.close()