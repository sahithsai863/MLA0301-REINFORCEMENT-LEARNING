import gymnasium as gym
import numpy as np
import random
from collections import deque

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam



env = gym.make("CartPole-v1")

state_size = int(env.observation_space.shape[0])
action_size = int(env.action_space.n)

print("State size:", state_size)
print("Action size:", action_size)



model = Sequential([
    Input(shape=(state_size,)),
    Dense(24, activation="relu"),
    Dense(24, activation="relu"),
    Dense(action_size, activation="linear")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse"
)

model.summary()



memory = deque(maxlen=2000)

gamma = 0.95

epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

episodes = 500
batch_size = 32



for episode in range(episodes):

    state, info = env.reset()

    total_reward = 0

    for step in range(100):



        if random.random() <= epsilon:

            action = env.action_space.sample()

        else:

            state_array = np.array(
                state,
                dtype=np.float32
            ).reshape(1, -1)

            q_values = model.predict(
                state_array,
                verbose=0
            )

            action = int(np.argmax(q_values[0]))


        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated


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

        total_reward += reward


        if len(memory) >= batch_size:

            batch = random.sample(
                memory,
                batch_size
            )

            states = np.array(
                [x[0] for x in batch],
                dtype=np.float32
            )

            actions = np.array(
                [x[1] for x in batch],
                dtype=np.int32
            )

            rewards = np.array(
                [x[2] for x in batch],
                dtype=np.float32
            )

            next_states = np.array(
                [x[3] for x in batch],
                dtype=np.float32
            )

            dones = np.array(
                [x[4] for x in batch],
                dtype=bool
            )


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

                    target = (
                        rewards[i]
                        + gamma * np.max(next_q[i])
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


    if epsilon > epsilon_min:

        epsilon *= epsilon_decay

        if epsilon < epsilon_min:
            epsilon = epsilon_min


    print(
        "Episode:",
        episode + 1,
        "/",
        episodes,
        "| Reward:",
        int(total_reward),
        "| Epsilon:",
        round(epsilon, 3)
    )



env.close()


print()
print("========================================")
print("       TRAINING COMPLETED")
print("========================================")


print()
print("Starting trained agent...")
print()


test_env = gym.make(
    "CartPole-v1",
    render_mode="human"
)

state, info = test_env.reset()

total_reward = 0

for step in range(500):


    state_array = np.array(
        state,
        dtype=np.float32
    ).reshape(1, -1)


    q_values = model.predict(
        state_array,
        verbose=0
    )


    action = int(
        np.argmax(q_values[0])
    )


    next_state, reward, terminated, truncated, info = test_env.step(
        action
    )

    done = terminated or truncated

    state = next_state

    total_reward += reward

    if done:
        break


test_env.close()


print()
print("========================================")
print("        TESTING COMPLETED")
print("========================================")

print(
    "Total Test Reward:",
    int(total_reward)
)

print(
    "Steps Survived:",
    step + 1
)

print()
print("DQN agent successfully completed testing!")
