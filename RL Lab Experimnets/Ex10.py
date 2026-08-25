import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

SIZE = 5
START = (0, 0)
GOAL = (4, 4)
BATTERY = 20

gamma = 0.95
alpha = 0.001

epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

episodes = 1000
batch_size = 32

memory = deque(maxlen=2000)

def normalize(state):
    return np.array([
        state[0] / (SIZE - 1),
        state[1] / (SIZE - 1),
        state[2] / BATTERY
    ], dtype=np.float32)

def step(state, action):
    row, col, battery = state

    if battery <= 0:
        return state, -20, True

    moves = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]

    new_row = row + moves[action][0]
    new_col = col + moves[action][1]
    new_battery = battery - 1

    if new_row < 0 or new_row >= SIZE or new_col < 0 or new_col >= SIZE:
        return state, -5, False

    next_state = (
        new_row,
        new_col,
        new_battery
    )

    if (new_row, new_col) == GOAL:
        return next_state, 100, True

    return next_state, -1, False

model = Sequential([
    Input(shape=(3,)),
    Dense(32, activation="relu"),
    Dense(32, activation="relu"),
    Dense(4, activation="linear")
])

model.compile(
    optimizer=Adam(learning_rate=alpha),
    loss="mse"
)

for episode in range(episodes):

    state = (
        START[0],
        START[1],
        BATTERY
    )

    for _ in range(50):

        if random.random() < epsilon:

            action = random.randint(0, 3)

        else:

            q_values = model.predict(
                normalize(state).reshape(1, -1),
                verbose=0
            )

            action = int(np.argmax(q_values[0]))

        next_state, reward, done = step(
            state,
            action
        )

        memory.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )

        state = next_state

        if len(memory) >= batch_size:

            batch = random.sample(
                memory,
                batch_size
            )

            states = np.array([
                normalize(x[0])
                for x in batch
            ])

            actions = np.array([
                x[1]
                for x in batch
            ])

            rewards = np.array([
                x[2]
                for x in batch
            ])

            next_states = np.array([
                normalize(x[3])
                for x in batch
            ])

            dones = np.array([
                x[4]
                for x in batch
            ])

            current_q = model.predict(
                states,
                verbose=0
            )

            next_q = model.predict(
                next_states,
                verbose=0
            )

            for i in range(batch_size):

                target = rewards[i]

                if not dones[i]:

                    target += (
                        gamma *
                        np.max(next_q[i])
                    )

                current_q[i][actions[i]] = target

            model.fit(
                states,
                current_q,
                epochs=1,
                verbose=0
            )

        if done:
            break

    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

    if (episode + 1) % 100 == 0:
        print(
            "Episode:",
            episode + 1,
            "| Epsilon:",
            round(epsilon, 3)
        )

state = (
    START[0],
    START[1],
    BATTERY
)

path = [
    (state[0], state[1])
]

total_reward = 0

for _ in range(50):

    q_values = model.predict(
        normalize(state).reshape(1, -1),
        verbose=0
    )

    action = int(
        np.argmax(q_values[0])
    )

    next_state, reward, done = step(
        state,
        action
    )

    total_reward += reward

    state = next_state

    path.append(
        (state[0], state[1])
    )

    if done:
        break

print()
print("DQN Training Completed")
print()
print("Start:", START)
print("Goal:", GOAL)
print("Initial Battery:", BATTERY)
print()
print("Optimal Delivery Route:")
print(path)
print()
print("Total Reward:", total_reward)
print("Remaining Battery:", state[2])
print("Steps:", len(path) - 1)
print(
    "Delivery Successful:",
    (state[0], state[1]) == GOAL
)
