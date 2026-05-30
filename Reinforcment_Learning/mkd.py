import numpy as np

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
    [np.nan,   -.25,      5,   -.25], # s1 - - -
    [np.nan,   -.25, np.nan,   -.25], # s2 - - -
    [  -.25,   -.25,   -.25, np.nan], # s3 - - -
    [  -.25,   -.25,   -.25,   -.25], # s4 - - -
    [     5,   -.25, np.nan,   -.25], # s5 - - -
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
total_reward = 0
position = np.random.randint(0, 9)
print(f"Started at position: {position} - {positions[position]}")

while True:
    action = int(input("Enter your position: "))
    while action not in actions[position]:
        action = int(input("Action not allowed: "))
    total_reward += rewards[position][action]
    if action == 0 or action == 1: position += 3 * (-1) ** (action+1)
    else: position += 1 * (-1) ** action
    print(f"Moved to position: {position} - {positions[position]}")
    if position == 2:
        print("You won!")
        break
        
print(f"Score: {total_reward}")