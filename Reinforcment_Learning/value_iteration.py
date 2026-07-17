import gymnasium as gym

import argparse
from time import sleep
from collections import defaultdict, Counter

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--iteration-exporation", type=int, default=100)

GAMMA = 0.9

class Agent:
    def __init__(self, env):
        self.env = env
        self.state, _ = env.reset()
        self.observation_space = 64
        self.action_space = 4

        self.state_value = defaultdict(float) # (s -> V(s))
        self.action_value = defaultdict(float) # ((s, a) -> Q(s, a))
        self.transtitions = defaultdict(Counter) # ((s, a) -> {s'->T(s, a, s')})
        self.rewards = defaultdict(float) # ((s, a, s') -> R(s, a, s'))

    def explore_env(self, iterations):
        for _ in range(iterations):
            action = self.env.action_space.sample()
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            self.transtitions[(self.state, action)][next_state] += 1
            self.rewards[(self.state, action, next_state)] = reward
            self.state = next_state
            if terminated or truncated:
                self.state, _ = self.env.reset()

    def update_values(self):
        for state in range(self.observation_space):
            actions_return = []
            for action in range(self.action_space):
                total = self.transtitions[(state, action)].total()
                value = 0
                for next_state, counter in self.transtitions[(state, action)].items():
                    value += counter / total * (self.rewards[(state, action, next_state)] + GAMMA * self.state_value[next_state])
                actions_return.append(value)
                self.action_value[(state, action)] = value
            # Is is possible to do one key lookup without the other composite key
            self.state_value[state] = max(actions_return)
 
    def select_best_action(self, state):
        best_action, best_reward = None, None
        for action in range(self.action_space):
            action_value = self.action_value[(state, action)]
            if best_reward is None or best_reward < action_value:
                best_reward = action_value
                best_action = action
        return best_action

    def test_policy(self, test_env, render=False):
        state, _ = test_env.reset()
        total_reward = 0
        while True:
            best_action = self.select_best_action(state)
            next_state, reward, terminated, truncated, _ = test_env.step(best_action)
            self.transtitions[(state, best_action)][next_state] += 1
            self.rewards[(state, best_action, next_state)]
            if render:
                sleep(.7)
            total_reward += reward
            if terminated or truncated:
                test_env.close()
                break
            state = next_state

        return total_reward

if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False)
    test_env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False)
    agent = Agent(env)

    total_rewards = 0
    best_reward = 0
    iteration = 0
    while True:
        agent.explore_env(100)
        agent.update_values()
        total_rewards = 0
        for i in range(20):
            rewards = agent.test_policy(test_env)
            if best_reward < rewards:
                best_reward = rewards
                print(f"Iteration {iteration} - Test {i}: best reward -> {best_reward}")
            total_rewards += rewards
        mean_reward = total_rewards / 20
        print("Iteration: ", iteration, " Mean Rewards: ", total_rewards / 20)
        
        if mean_reward > 0.8:
            break
        total_rewards = 0
        iteration += 1

    demo_env = gym.make("FrozenLake-v1", map_name="8x8", render_mode="human", is_slippery=False)
    agent.test_policy(demo_env, render=True)