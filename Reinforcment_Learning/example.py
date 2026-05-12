import gymnasium as gym

env = gym.make("Acrobot-v1", render_mode="human")
obs, info = env.reset()

episode_over = False
totals = 0

while not episode_over:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    totals += reward
    episode_over = terminated or truncated
    
print(totals)
env.close()