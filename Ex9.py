import numpy as np
import random

size = 5
start = (0, 0)
goal = (4, 4)

episodes = 1000
alpha = 0.1
gamma = 0.9
epsilon = 0.1

actions = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1)
]

def step(state, action):
    row, col = state

    next_row = row + actions[action][0]
    next_col = col + actions[action][1]

    if next_row < 0 or next_row >= size or next_col < 0 or next_col >= size:
        return state, -5, False

    next_state = (next_row, next_col)

    if next_state == goal:
        return next_state, 100, True

    return next_state, -1, False

def choose_action(Q, state):
    if random.random() < epsilon:
        return random.randint(0, 3)

    return int(np.argmax(Q[state[0], state[1]]))

def td_zero():
    V = np.zeros((size, size))

    for _ in range(episodes):
        state = start

        for _ in range(100):
            action = choose_action(
                np.zeros((size, size, 4)),
                state
            )

            next_state, reward, done = step(state, action)

            target = reward

            if not done:
                target += gamma * V[next_state[0], next_state[1]]

            V[state[0], state[1]] += alpha * (
                target - V[state[0], state[1]]
            )

            state = next_state

            if done:
                break

    return V

def sarsa():
    Q = np.zeros((size, size, 4))

    for _ in range(episodes):
        state = start
        action = choose_action(Q, state)

        for _ in range(100):
            next_state, reward, done = step(state, action)

            if done:
                target = reward
                Q[state[0], state[1], action] += alpha * (
                    target - Q[state[0], state[1], action]
                )
                break

            next_action = choose_action(Q, next_state)

            target = reward + gamma * Q[
                next_state[0],
                next_state[1],
                next_action
            ]

            Q[state[0], state[1], action] += alpha * (
                target - Q[state[0], state[1], action]
            )

            state = next_state
            action = next_action

    return Q

def q_learning():
    Q = np.zeros((size, size, 4))

    for _ in range(episodes):
        state = start

        for _ in range(100):
            action = choose_action(Q, state)

            next_state, reward, done = step(
                state,
                action
            )

            target = reward

            if not done:
                target += gamma * np.max(
                    Q[next_state[0], next_state[1]]
                )

            Q[state[0], state[1], action] += alpha * (
                target - Q[state[0], state[1], action]
            )

            state = next_state

            if done:
                break

    return Q

def get_path(Q):
    state = start
    path = [state]

    for _ in range(100):
        if state == goal:
            break

        action = int(
            np.argmax(Q[state[0], state[1]])
        )

        next_state, _, done = step(
            state,
            action
        )

        if next_state == state:
            break

        path.append(next_state)
        state = next_state

        if done:
            break

    return path

V = td_zero()
SARSA = sarsa()
Q = q_learning()

sarsa_path = get_path(SARSA)
qlearning_path = get_path(Q)

print("TD(0) State Value at Start:",
      round(V[start[0], start[1]], 2))

print("\nSARSA Optimal Path:")
print(sarsa_path)

print("\nQ-Learning Optimal Path:")
print(qlearning_path)

print("\nGoal Reached by SARSA:",
      sarsa_path[-1] == goal)

print("Goal Reached by Q-Learning:",
      qlearning_path[-1] == goal)
