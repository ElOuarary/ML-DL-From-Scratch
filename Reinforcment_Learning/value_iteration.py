import gymnasium as gym

import argparse
from time import sleep
from collections import defaultdict, Counter

GAMMA = 0.9


class Agent:
    def __init__(self, env):
        self.env = env
        self.state, _ = env.reset()
        self.observation_space = 64
        self.action_space = 4

        self.state_value = defaultdict(float)  # (s -> V(s))
        self.action_value = defaultdict(float)  # ((s, a) -> Q(s, a))
        self.transitions = defaultdict(Counter)  # ((s, a) -> {s'->T(s, a, s')})
        self.rewards = defaultdict(float)  # ((s, a, s') -> R(s, a, s'))

    def explore_env(self, iterations):
        for _ in range(iterations):
            action = self.env.action_space.sample()
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            self.transitions[(self.state, action)][next_state] += 1
            self.rewards[(self.state, action, next_state)] = reward
            self.state = next_state
            if terminated or truncated:
                self.state, _ = self.env.reset()

    def update_values(self):
        for state in range(self.observation_space):
            actions_return = []
            for action in range(self.action_space):
                total = self.transitions[(state, action)].total()
                value = 0
                for next_state, counter in self.transitions[(state, action)].items():
                    value += (
                        counter
                        / total
                        * (
                            self.rewards[(state, action, next_state)]
                            + GAMMA * self.state_value[next_state]
                        )
                    )
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
            self.transitions[(state, best_action)][next_state] += 1
            self.rewards[(state, best_action, next_state)] = reward
            if render:
                sleep(0.7)
            total_reward += reward
            if terminated or truncated:
                break
            state = next_state

        return total_reward


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--exploration", type=int, default=100)

    args = arg_parser.parse_args()
    EXPLORATION = args.exploration

    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False)
    test_env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False)
    demo_env = gym.make(
        "FrozenLake-v1", map_name="8x8", render_mode="human", is_slippery=False
    )
    agent = Agent(env)

    total_rewards = 0
    best_reward = 0
    iteration = 0
    try:
        while True:
            agent.explore_env(EXPLORATION)
            agent.update_values()
            total_rewards = 0
            for i in range(20):
                rewards = agent.test_policy(test_env)
                if best_reward < rewards:
                    best_reward = rewards
                    print(
                        f"Iteration {iteration} - Test {i}: best reward -> {best_reward}"
                    )
                total_rewards += rewards
            mean_reward = total_rewards / 20
            print("Iteration: ", iteration, " Mean Rewards: ", mean_reward)

            if mean_reward > 0.8:
                break

            iteration += 1

        agent.test_policy(demo_env, render=True)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        test_env.close()
        demo_env.close()
