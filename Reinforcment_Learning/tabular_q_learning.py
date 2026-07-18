import gymnasium as gym
import numpy as np

import pprint

from collections import defaultdict
from time import sleep

ALPHA = 0.1
GAMMA = 0.9
TEST_ITERATION = 20

class Agent:
    def __init__(self, env):
        self.env = env
        self.state, _ = env.reset()
        self.action_space = 4
        self.q_values = defaultdict(float)
    
    def explore_env(self, n):
        for _ in range(n):
            old_state = self.state
            action = self.env.action_space.sample()
            next_state, reward,terminated, truncated, _ = self.env.step(action)
            self.update_q_values(old_state, action, reward, next_state)
            if terminated or truncated:
                self.state, _ = self.env.reset()
                continue
            self.state = next_state

    def update_q_values(self, state, action, reward, next_state):
        _, future_discouted_reward = self.select_best_action(next_state)
        self.q_values[(state, action)] = (1 - ALPHA) * self.q_values[(state, action)] + ALPHA * (reward + GAMMA * future_discouted_reward)

    def select_best_action(self, state):
        best_action, best_reward = None, None
        for action in range(self.action_space):
            q_value = self.q_values[(state, action)]
            if best_reward is None or best_reward < q_value:
                best_action = action
                best_reward = q_value
        return best_action, best_reward

    def test_env(self, env, render=False):
        state, _  = env.reset()
        total_reward = 0
        while True:
            best_action, _ = self.select_best_action(state)
            next_state, reward, terminated, truncated, _ = env.step(best_action)
            if render:
                sleep(0.7)
            total_reward += reward
            state = next_state
            if terminated or truncated:
                break
        return total_reward

if __name__ == "__main__":
    env = gym.make("CliffWalking-v1")
    test_env = gym.make("CliffWalking-v1")
    demo_env = gym.make("CliffWalking-v1", render_mode="human")
    agent = Agent(env)

    i = 1
    top_reward = 0
    while True:
        agent.explore_env(10_000)
        total_rewards = 0
        for _ in range(20):
            reward = agent.test_env(test_env)
            if reward > top_reward:
                print(f"Updating top reward {top_reward} -> {reward}")
                top_reward = reward
                total_rewards += reward
        mean_reward = total_rewards / 20
        print(f"Iteration {i} - Mean Reward: {mean_reward}")
        if mean_reward > -15:
            print("Problem Solved!")
            break

        i += 1

    agent.test_env(demo_env, True)