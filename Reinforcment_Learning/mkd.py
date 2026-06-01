import numpy as np
import matplotlib.pyplot as plt

transition_probabilities = np.array([ # (s, a, s')
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # move up impossible
        [0, 0, 0, 1, 0, 0, 0, 0, 0], # move down to s3
        [0, 1, 0, 0, 0, 0, 0, 0, 0], # move righ to s1
        [0, 0, 0, 0, 0, 0, 0, 0, 0]  # move left impossible
    ], # left upper s0
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # move up impossitble
        [0, 0, 0, 0, 1, 0, 0, 0, 0], # move down to s4
        [0, 0, 1, 0, 0, 0, 0, 0, 0], # move right to s2
        [1, 0, 0, 0, 0, 0, 0, 0, 0]  # move left to s0
    ], # middle upper s1
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # move up impossible
        [0, 0, 0, 0, 0, 1, 0, 0, 0], # move down to s5
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # move right impossible
        [0, 1, 0, 0, 0, 0, 0, 0, 0]  # move left to s1
    ], # right upper s2
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0], # move up to s0
        [0, 0, 0, 0, 0, 0, 1, 0, 0], # move down to s6
        [0, 0, 0, 0, 1, 0, 0, 0, 0], # move right to s4
        [0, 0, 0, 0, 0, 0, 0, 0, 0]  # move left impossible
    ], # left middle s3
    [
        [0, 1, 0, 0, 0, 0, 0, 0, 0], # move up to s1
        [0, 0, 0, 0, 0, 0, 0, 1, 0], # move down to s7
        [0, 0, 0, 0, 0, 1, 0, 0, 0], # move right s5
        [0, 0, 0, 1, 0, 0, 0, 0, 0]  # move left to s3
    ], # middle s4
    [
        [0, 0, 1, 0, 0, 0, 0, 0, 0], # move up to s2
        [0, 0, 0, 0, 0, 0, 0, 0, 1], # move down to s8
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # move right impossible
        [0, 0, 0, 0, 1, 0, 0, 0, 0]  # move left to s4
    ], # right middle s5
    [
        [0, 0, 0, 1, 0, 0, 0, 0, 0], # move up to s3
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # move down impossible
        [0, 0, 0, 0, 0, 0, 0, 1, 0], # move right to s7
        [0, 0, 0, 0, 0, 0, 0, 0, 0]  # move left to impossible
    ], # right middle s6
    [
        [0, 0, 0, 0, 1, 0, 0, 0, 0], # move up to s4
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # move down impossible
        [0, 0, 0, 0, 0, 0, 0, 0, 1], # move right s8
        [0, 0, 0, 0, 0, 0, 1, 0, 0]  # move left to s6
    ], # right middle s7
    [
        [0, 0, 0, 0, 0, 1, 0, 0, 0], # move up to s5
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # move down impossible
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # move right impossible
        [0, 0, 0, 0, 0, 0, 0, 1, 0]  # move left to s4
    ], # right middle s8
])

rewards = np.array([
    [np.nan,   -.25,   -.25, np.nan], # s0 reward per action
    [np.nan,   -.25,      1,   -.25], # s1 - - -
    [np.nan,   -.25, np.nan,   -.25], # s2 - - -
    [  -.25,   -.25,   -.25, np.nan], # s3 - - -
    [  -.25,   -.25,   -.25,   -.25], # s4 - - -
    [     1,   -.25, np.nan,   -.25], # s5 - - -
    [  -.25, np.nan,   -.25, np.nan], # s6 - - -
    [  -.25, np.nan,   -.25,   -.25], # s7 - - -
    [  -.25, np.nan, np.nan,   -.25]  # s8 - - -
])

actions = (
    [1, 2],       # s0 actions
    [1, 2, 3],    # s1 -
    [1, 3],       # s2 -
    [0, 1, 2],    # s3 -
    [0, 1, 2, 3], # s4 -
    [0, 1, 3],    # s5 -
    [0, 2],       # s6 -
    [0, 2, 3],    # s7 -
    [0, 3]        # s8 -
)

positions = {
    0:  "upper left", 1: "upper middle", 2:  "upper right",
    3: "middle left", 4:       "middle", 5: "middle right",
    6:  "lower left", 7: "lower middle", 8:  "lower right"
}

V_MDP = np.zeros(9)

Q_VALUE = np.full((9, 4), -np.inf)
for state, action in  enumerate(actions):
    Q_VALUE[state][action] = 0

TD_V = np.zeros(9)
TD_Q = np.full((9, 4), -np.inf)
for state, action in enumerate(actions):
    TD_Q[state][action] = 0

gamma = .95
decay = 0.005
alpha = 0.05

for i in range(0, 10_000):
    if i >= 100:
        V_MDP_prev = V_MDP.copy()
        Q_VALUE_prev = Q_VALUE.copy()
    TD_V_prev = TD_V.copy()
    TD_Q_prev = TD_Q.copy()
    for s in range(9):
        
        if i >= 100:
            V_MDP[s] = np.max([
                np.sum(rewards[s][action] + gamma * V_MDP_prev[np.argmax(transition_probabilities[s][action])])
                for action in actions[s]
            ])
        
            for action in actions[s]:
                Q_VALUE[s][action] = np.sum([
                    transition_probabilities[s][action][next_s] * (rewards[s][action] + gamma * Q_VALUE_prev[next_s].max())
                    for next_s in range(9)
                ])
        
        random_action = np.random.choice(actions[s])
        next_state = np.argmax(transition_probabilities[s][random_action])
        
        alpha = alpha / (1 + i * decay)
            
        TD_V[s] += alpha * (rewards[s][random_action] + gamma * TD_V_prev[next_state] - TD_V_prev[s])
        TD_Q[s][random_action] += alpha * (rewards[s][random_action] + gamma * TD_Q_prev[next_state].max() - TD_Q_prev[s][random_action])

print(f"Ieration number: {i}")
print(V_MDP.reshape(3, 3))
print(Q_VALUE)
print(TD_V)
print(TD_Q)
        
starting_state = np.random.randint(0, 9)
state = starting_state
print(f"Starting State {state} - {positions[state]}")

print("The Otpimal State Method".center(50, "="))
while state != 2:
    print(f"Current State {state} - {positions[state]}")
    best_action = max(
        actions[state],
        key=lambda action: np.sum(rewards[state][action] + gamma * V_MDP[np.argmax(transition_probabilities[state][action])])
    )
    next_state = np.argmax(transition_probabilities[state][best_action])
    state = next_state
    print(f"Moving to {positions[next_state]}")
    
print()

state = starting_state
print("The Optimal Q-value Mehtod".center(50, "="))
while state != 2:
    print(f"Current state {state} - {positions[state]}")
    best_action = Q_VALUE[state].argmax()
    next_state = np.argmax(transition_probabilities[state][best_action])
    state = next_state
    print(f"Moving to {positions[next_state]}")

print()
    
state = starting_state
print("The TD For Optimal State Value".center(50, "="))
while state != 2:
    print(f"Current state {state} - {positions[state]}")
    best_action = max(
        actions[state],
        key=lambda action: np.sum(rewards[state][action] + gamma * TD_V[np.argmax(transition_probabilities[state][action])])
    )
    state = next_state
    print(f"Moving to {positions[next_state]}")

print()    

state = starting_state
print("The TD For Optimal Q-value Mehtod".center(50, "="))
while state != 2:
    print(f"Current state {state} - {positions[state]}")
    best_action = TD_Q[state].argmax()
    next_state = np.argmax(transition_probabilities[state][best_action])
    state = next_state
    print(f"Moving to {positions[next_state]}")