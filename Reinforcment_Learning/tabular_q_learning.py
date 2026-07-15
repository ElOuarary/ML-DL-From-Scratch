import gymnasium as gym

from collections import defaultdict

GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20

class Agent:
    def __init__(self):
        self.env = gym.make("FrozenLake-v1", is_slippery=False)
        self.state, _ = self.env.reset()
        self.q_values = defaultdict(float)

    def sample_env(self):
        action = self.env.action_space.sample()
        old_state = self.state
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        self.state, _ = self.env.reset() if terminated or truncated else next_state, None
        return old_state, action, reward, next_state

    def best_value_and_action(self, state):
        best_action, best_value = None, None
        for action in range(4):
            q_value = self.q_values[(state, action)]
            if best_action is None or best_value < q_value:
                best_action = action
                best_value = q_value
        return best_action, best_value
    
    def update_value(self, state, action, reward, next_state):
        best_action, _ = self.best_value_and_action(next_state)
        self.q_values[(state, action)] = (1 - ALPHA) * self.q_values[(state, action)] + ALPHA * (reward + GAMMA * self.q_values[(next_state, best_action)])

    def test_episode(self, env):
        total_reward = 0
        state, _ = env.reset()
        while True:
            best_action, _ = self.best_value_and_action(state)
            next_state, reward, terminated, truncated, _ = env.step(best_action)
            total_reward += reward
            if terminated or truncated:
                break
            state = next_state
        return total_reward
    
if __name__ == "__main__":
    test_env = gym.make("FrozenLake-v1")
    agent = Agent()
    
    iteration_no = 0
    best_reward = 0
    while True:
        iteration_no += 1
        state, action, reward, next_state = agent.sample_env()
        print(state, action, reward, next_state)
        agent.update_value(state, action, reward, next_state)

        reward = 0
        for i in range(TEST_EPISODES):
            reward += agent.test_episode(test_env)
        reward /= TEST_EPISODES

        if best_reward < reward:
            print(f"Best Reward Update in iteration {i} -> {reward}")
        if reward > 0.8:
            print(f"Solved in iteration {i} with a reward of {reward}")
            break
    agent.env.close()
    test_env.close()