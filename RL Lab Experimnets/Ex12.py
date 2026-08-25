import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

GRID_SIZE = 5
START = (0, 0)
GOAL = (4, 4)

OBSTACLES = {(1, 1), (2, 1), (2, 3), (3, 3)}

class ParkingEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.position = START
        return self.state()

    def state(self):
        state = np.zeros(GRID_SIZE * GRID_SIZE, dtype=np.float32)
        r, c = self.position
        state[r * GRID_SIZE + c] = 1

        gr, gc = GOAL
        state[gr * GRID_SIZE + gc] = 2

        for r, c in OBSTACLES:
            state[r * GRID_SIZE + c] = -1

        return state

    def step(self, action):
        r, c = self.position

        if action == 0:
            nr, nc = r - 1, c
        elif action == 1:
            nr, nc = r + 1, c
        elif action == 2:
            nr, nc = r, c - 1
        else:
            nr, nc = r, c + 1

        if nr < 0 or nr >= GRID_SIZE or nc < 0 or nc >= GRID_SIZE:
            return self.state(), -10, True

        if (nr, nc) in OBSTACLES:
            return self.state(), -50, True

        old_distance = abs(r - GOAL[0]) + abs(c - GOAL[1])
        new_distance = abs(nr - GOAL[0]) + abs(nc - GOAL[1])

        self.position = (nr, nc)

        if self.position == GOAL:
            return self.state(), 100, True

        if new_distance < old_distance:
            reward = 5
        else:
            reward = -2

        return self.state(), reward - 1, False


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(25, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.network(x)


env = ParkingEnv()
model = PolicyNetwork()

optimizer = optim.Adam(model.parameters(), lr=0.01)

gamma = 0.99
episodes = 1000

rewards_history = []

for episode in range(episodes):

    state = env.reset()

    log_probs = []
    rewards = []

    for step in range(50):

        state_tensor = torch.tensor(state, dtype=torch.float32)

        probabilities = model(state_tensor)

        distribution = torch.distributions.Categorical(probabilities)

        action = distribution.sample()

        next_state, reward, done = env.step(action.item())

        log_probs.append(distribution.log_prob(action))
        rewards.append(reward)

        state = next_state

        if done:
            break

    returns = []
    G = 0

    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32)

    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    loss = 0

    for log_prob, G in zip(log_probs, returns):
        loss += -log_prob * G

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_reward = sum(rewards)
    rewards_history.append(total_reward)

    if (episode + 1) % 100 == 0:
        print(
            "Episode:",
            episode + 1,
            "Reward:",
            total_reward,
            "Steps:",
            len(rewards)
        )


print("\nTraining completed")


def evaluate():
    state = env.reset()

    path = [env.position]

    for step in range(50):

        state_tensor = torch.tensor(state, dtype=torch.float32)

        with torch.no_grad():
            probabilities = model(state_tensor)

        action = torch.argmax(probabilities).item()

        state, reward, done = env.step(action)

        path.append(env.position)

        if done:
            break

    return path, env.position == GOAL


path, success = evaluate()

print("\nLearned Parking Path:")
print(path)

if success:
    print("Parking Successful")
else:
    print("Parking Failed")
