import gymnasium as gym
import numpy as np

import argparse
from collections import defaultdict, Counter

parser = argparse.ArgumentParser()
parser.add_argument("--gamma", default=0.9, type=float)
parser.add_argument("--test_episode", default=20, type=int)

args = parser.parse_args()
GAMMA = args.gamma
TEST_EPISODE = args.test_episode

class Agent:
    def __init__(self):
        self.env = gym.make("FrozenLake-v1", render_mode="human")
        self.state, _ = self.env.reset()
        self.rewards = defaultdict(float)
        self.transitions = defaultdict(Counter)
        self.values = defaultdict(float)
        
    def play_n_random(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            self.rewards[(self.state, action, next_state)] = reward
            self.transitions[(self.state, action)][next_state] += 1
            if terminated or truncated:
                self.state, _ = env.reset()
            else:
                self.state = next_state
            
    def calc_action_value(self, state, action):
        count = self.transitions[(state, action)]
        total = sum(count.values())
        value_action = 0
        for target_state, count in self.transitions[(state, action)].items():
            value_action += (count / total) * (self.rewards[(state, action, target_state)] + GAMMA * self.values[target_state])
        return value_action
    
    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(4):
            value_action = self.calc_action_value(state, action)
            if best_action is None or best_value < value_action:
                best_action = action
                best_value = value_action
        return best_action
    
    def play_episode(self, env):
        state, _ = env.reset()
        total_reward = 0
        while True:
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            self.rewards[(state, action, next_state)] = reward
            self.transitions[(state, action)][next_state] += 1
            total_reward += reward
            if terminated or truncated:
                break
            state = next_state
        return total_reward

    def value_iteration(self):
        for state in range(16):
            state_values = [
                self.calc_action_value(state, action)
                for action in range(4)
            ]
            self.values[state] = max(state_values)
        
if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", render_mode="human")
    agent = Agent()
    iter_no = 0
    best_reward = -np.inf
    while True:
        iter_no += 1
        agent.play_n_random(100)
        agent.value_iteration()
        
        reward = 0
        for _ in range(TEST_EPISODE):
            reward += agent.play_episode(env)
        reward /= TEST_EPISODE
        if reward > best_reward:
            print(f"Best reward update: {best_reward} -> {reward}")
            best_reward = reward
        if reward > 0.8:
            print(f"Solved {iter_no} in iteration")
            break
    env.close()