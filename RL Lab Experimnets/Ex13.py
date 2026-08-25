import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

GRID_SIZE = 5
MAX_STEPS = 50

class RobotArmEnv:

    def __init__(self):
        self.reset()

    def reset(self):
        self.robot = [0, 0]
        self.object = [2, 2]
        self.target = [4, 4]
        self.picked = False
        self.steps = 0
        return self.get_state()

    def get_state(self):
        state = np.array([
            self.robot[0] / 4,
            self.robot[1] / 4,
            self.object[0] / 4,
            self.object[1] / 4,
            self.target[0] / 4,
            self.target[1] / 4,
            float(self.picked)
        ], dtype=np.float32)

        return state

    def step(self, action):

        self.steps += 1
        old_distance = 0

        if not self.picked:
            old_distance = abs(
                self.robot[0] - self.object[0]
            ) + abs(
                self.robot[1] - self.object[1]
            )
        else:
            old_distance = abs(
                self.robot[0] - self.target[0]
            ) + abs(
                self.robot[1] - self.target[1]
            )

        if action == 0:
            self.robot[1] = min(
                GRID_SIZE - 1,
                self.robot[1] + 1
            )

        elif action == 1:
            self.robot[1] = max(
                0,
                self.robot[1] - 1
            )

        elif action == 2:
            self.robot[0] = max(
                0,
                self.robot[0] - 1
            )

        elif action == 3:
            self.robot[0] = min(
                GRID_SIZE - 1,
                self.robot[0] + 1
            )

        elif action == 4:

            if not self.picked:

                if self.robot == self.object:
                    self.picked = True
                    reward = 20
                else:
                    reward = -20

            else:

                if self.robot == self.target:
                    reward = 100
                    return self.get_state(), reward, True
                else:
                    reward = -20

            return (
                self.get_state(),
                reward,
                self.steps >= MAX_STEPS
            )

        if not self.picked:

            new_distance = abs(
                self.robot[0] - self.object[0]
            ) + abs(
                self.robot[1] - self.object[1]
            )

        else:

            new_distance = abs(
                self.robot[0] - self.target[0]
            ) + abs(
                self.robot[1] - self.target[1]
            )

        if new_distance < old_distance:
            reward = 5
        else:
            reward = -2

        reward -= 1

        done = self.steps >= MAX_STEPS

        return self.get_state(), reward, done


class PolicyNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.network(state)


env = RobotArmEnv()

model = PolicyNetwork()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)

gamma = 0.99

episodes = 1000

reward_history = []

success_history = []


for episode in range(episodes):

    state = env.reset()

    log_probs = []
    rewards = []

    done = False

    while not done:

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        )

        probabilities = model(state_tensor)

        distribution = torch.distributions.Categorical(
            probabilities
        )

        action = distribution.sample()

        next_state, reward, done = env.step(
            action.item()
        )

        log_probs.append(
            distribution.log_prob(action)
        )

        rewards.append(reward)

        state = next_state

    returns = []

    G = 0

    for reward in reversed(rewards):

        G = reward + gamma * G

        returns.insert(0, G)

    returns = torch.tensor(
        returns,
        dtype=torch.float32
    )

    if len(returns) > 1:

        returns = (
            returns - returns.mean()
        ) / (
            returns.std() + 1e-8
        )

    loss = 0

    for log_prob, G in zip(
        log_probs,
        returns
    ):

        loss += -log_prob * G

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    total_reward = sum(rewards)

    reward_history.append(total_reward)

    success = (
        env.robot == env.target
        and env.picked
    )

    success_history.append(
        int(success)
    )

    if (episode + 1) % 100 == 0:

        success_rate = (
            np.mean(success_history[-100:]) * 100
        )

        print(
            "Episode:",
            episode + 1,
            "Reward:",
            round(total_reward, 2),
            "Success Rate:",
            round(success_rate, 2),
            "%"
        )


print("\nTraining Completed")


def evaluate():

    state = env.reset()

    path = [tuple(env.robot)]

    actions = []

    for step in range(MAX_STEPS):

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        )

        with torch.no_grad():

            probabilities = model(
                state_tensor
            )

        action = torch.argmax(
            probabilities
        ).item()

        actions.append(action)

        state, reward, done = env.step(
            action
        )

        path.append(tuple(env.robot))

        if done:
            break

    return path, actions, env.picked, env.robot == env.target


path, actions, picked, placed = evaluate()

print("\nRobot Path:")
print(path)

print("\nActions:")
print(actions)

print("\nObject Picked:", picked)

print("Object Placed:", placed)

if placed:
    print("\nPick-and-Place Successful")
else:
    print("\nPick-and-Place Failed")
