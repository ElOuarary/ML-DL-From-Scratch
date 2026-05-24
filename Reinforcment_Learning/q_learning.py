import numpy as np

transition_probabilities = [ #shape [s, a, s']
    [[.7, .3, .0], [1.0, .0, .0], [.8, .2, 0]],
    [[.0, 1.0, .0], None, [.0, .0, 1.0]],
    [None, [.8, .1, .1], None]
]

rewards = [ # shape [s, a, s']
    [[10, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 0, -50]],
    [[0, 0, 0], [40, 0, 0], [0, 0, 0]]
]

possible_actions = [[0, 1, 2], [0, 2], [1]]

def step(state, action):
    probas = transition_probabilities[state][action]
    next_state = np.random.choice([0, 1, 2])
    reward = rewards[state][action][next_state]
    return next_state, reward

def exploration_policy(state):
    return np.random.choice(possible_actions[state])

alpha0 = 0.05 # learning rate
decay = 0.005 # learning rate decay 
gamma = 0.9 # discout factor
state = 0 # initial state

for iteration in range(10_000):
    action = exploration_policy(state)
    next_state, reward = step(state, action)
    next_value = Q_values[next_state].max()
    alpha = 1 / (1+ iteration * decay)
    Q_values[state, action] *= 1 - alpha
    Q_values[state, action] += alpha * (reward + gamma * next_value)
    state = next_state
    