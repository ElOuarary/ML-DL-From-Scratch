import numpy as np

transition_probabilities = [ #shape [s', a, s]
    [[.7, .3, .0], [1.0, .0, .0], [.8, .2, 0]],
    [[.0, 1.0, .0], None, [.0, .0, 1.0]],
    [None, [.8, .1, .1], None]
]

rewards = [ # shape [s', a, s]
    [[10, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 0, -50]],
    [[0, 0, 0], [40, 0, 0], [0, 0, 0]]
]

possible_actions = [[0, 1, 2], [0, 2], [1]]

V = np.zeros(3)

gamma = .9

for _ in range(3):
    V_previous = V.copy()
    for s in range(3):
            V[s] = max(
                sum(transition_probabilities[s][a][sp] * (rewards[s][a][sp] + gamma * V_previous[sp]) for sp in range(3))
                for a in possible_actions[s]
            )

print(V)

Q_values = np.full((3,3), -np.inf)

for state, actions in enumerate(possible_actions):
    Q_values[state, actions] = 0 # For possible actions
    

for iteration in range(50):
    Q_prev = Q_values.copy()
    for s in range(3):
        for a in possible_actions[s]:
            Q_values[s, a] = np.sum([
                transition_probabilities[s][a][sp] * (rewards[s][a][sp] + gamma * Q_prev[sp].max()) for sp in range(3)
            ])
            
print(Q_values.argmax(1))


gamma = .95

for iteration in range(50):
    Q_prev = Q_values.copy()
    for s in range(3):
        for a in possible_actions[s]:
            Q_values[s, a] = np.sum([
                transition_probabilities[s][a][sp] * (rewards[s][a][sp] + gamma * Q_prev[sp].max()) for sp in range(3)
            ])
            
print(Q_values.argmax(1))