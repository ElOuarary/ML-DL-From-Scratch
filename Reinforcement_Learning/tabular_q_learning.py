import gymnasium as gym

import argparse

from collections import defaultdict
from time import sleep


class Agent:
    def __init__(self, env, alpha, gamma):
        self.env = env
        self.state, _ = env.reset()

        self.alpha = alpha
        self.gamma = gamma

        self.action_space = self.env.action_space.n
        self.q_values = defaultdict(float)

    def explore_env(self, n):
        for _ in range(n):
            old_state = self.state
            action = self.env.action_space.sample()
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            self.update_q_values(old_state, action, reward, next_state)
            if terminated or truncated:
                self.state, _ = self.env.reset()
                continue
            self.state = next_state

    def update_q_values(self, state, action, reward, next_state):
        _, future_discouted_reward = self.select_best_action(next_state)
        self.q_values[(state, action)] = (1 - self.alpha) * self.q_values[
            (state, action)
        ] + self.alpha * (reward + self.gamma * future_discouted_reward)

    def select_best_action(self, state):
        best_action, best_reward = None, None
        for action in range(self.action_space):
            q_value = self.q_values[(state, action)]
            if best_reward is None or best_reward < q_value:
                best_action = action
                best_reward = q_value
        return best_action, best_reward

    def test_env(self, env, render=False):
        state, _ = env.reset()
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

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--alpha", type=float, default=0.1)
    arg_parser.add_argument("--gamma", type=float, default=0.9)
    arg_parser.add_argument("--exploration", type=int, default=1000)
    arg_parser.add_argument("--testing", type=int, default=20)
    arg_parser.add_argument("--render", type=bool, default=False)

    args = arg_parser.parse_args()
    ALPHA = args.alpha
    GAMMA = args.gamma
    EXPLORATION = args.exploration
    TESTING = args.testing
    render = "human" if args.render else None

    env = gym.make("CliffWalking-v1")
    test_env = gym.make("CliffWalking-v1")
    demo_env = gym.make("CliffWalking-v1", render_mode=render)
    agent = Agent(env, alpha=ALPHA, gamma=GAMMA)

    i = 1
    top_reward = 0
    try:
        while True:
            agent.explore_env(EXPLORATION)
            total_rewards = 0
            for _ in range(TESTING):
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
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        test_env.close()
    if render:
        try:
            agent.test_env(demo_env, True)
        except Exception:
            pass
        finally:
            demo_env.close()


if __name__ == "__main__":
    main()