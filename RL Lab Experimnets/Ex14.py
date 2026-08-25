import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import multiprocessing as mp

FLOORS = 5
ELEVATORS = 2
MAX_STEPS = 50


class ElevatorEnv:

    def __init__(self):
        self.reset()

    def reset(self):
        self.elevator_positions = [0, 4]
        self.requests = []
        self.total_wait = 0
        self.served = 0
        self.steps = 0
        return self.get_state()

    def get_state(self):
        state = np.zeros(10, dtype=np.float32)

        state[self.elevator_positions[0]] = 1
        state[5 + self.elevator_positions[1]] = 1

        for floor in self.requests:
            state[floor] += 0.2

        return state

    def step(self, action):

        self.steps += 1

        if len(self.requests) == 0:
            floor = random.randint(0, FLOORS - 1)
            self.requests.append(floor)

        target = self.requests[0]

        elevator = action

        distance = abs(self.elevator_positions[elevator] - target)

        if self.elevator_positions[elevator] < target:
            self.elevator_positions[elevator] += 1
        elif self.elevator_positions[elevator] > target:
            self.elevator_positions[elevator] -= 1

        reward = -distance * 0.5

        self.total_wait += distance

        if self.elevator_positions[elevator] == target:

            self.requests.pop(0)

            self.served += 1

            reward += 20

        if distance > 3:
            reward -= 5

        reward -= 1

        done = self.steps >= MAX_STEPS

        return self.get_state(), reward, done


class ActorCritic(nn.Module):

    def __init__(self):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.actor = nn.Linear(64, ELEVATORS)

        self.critic = nn.Linear(64, 1)

    def forward(self, state):

        x = self.shared(state)

        policy = torch.softmax(self.actor(x), dim=-1)

        value = self.critic(x)

        return policy, value


def train_a2c(episodes=500):

    env = ElevatorEnv()

    model = ActorCritic()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    gamma = 0.99

    reward_history = []

    for episode in range(episodes):

        state = env.reset()

        log_probs = []
        values = []
        rewards = []

        done = False

        while not done:

            state_tensor = torch.tensor(
                state,
                dtype=torch.float32
            )

            policy, value = model(state_tensor)

            distribution = torch.distributions.Categorical(
                policy
            )

            action = distribution.sample()

            next_state, reward, done = env.step(
                action.item()
            )

            log_probs.append(
                distribution.log_prob(action)
            )

            values.append(value)

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

        values = torch.cat(values).squeeze()

        advantages = returns - values.detach()

        actor_loss = 0

        for log_prob, advantage in zip(
            log_probs,
            advantages
        ):

            actor_loss += -log_prob * advantage

        critic_loss = nn.functional.mse_loss(
            values,
            returns
        )

        loss = actor_loss + 0.5 * critic_loss

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_reward = sum(rewards)

        reward_history.append(total_reward)

        if (episode + 1) % 50 == 0:

            print(
                "A2C Episode:",
                episode + 1,
                "Reward:",
                round(total_reward, 2),
                "Waiting:",
                env.total_wait
            )

    return model, reward_history


def train_worker(worker_id, episodes=100):

    env = ElevatorEnv()

    model = ActorCritic()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    gamma = 0.99

    rewards_history = []

    for episode in range(episodes):

        state = env.reset()

        log_probs = []
        values = []
        rewards = []

        done = False

        while not done:

            state_tensor = torch.tensor(
                state,
                dtype=torch.float32
            )

            policy, value = model(state_tensor)

            distribution = torch.distributions.Categorical(
                policy
            )

            action = distribution.sample()

            next_state, reward, done = env.step(
                action.item()
            )

            log_probs.append(
                distribution.log_prob(action)
            )

            values.append(value)

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

        values = torch.cat(values).squeeze()

        advantages = returns - values.detach()

        actor_loss = 0

        for log_prob, advantage in zip(
            log_probs,
            advantages
        ):

            actor_loss += -log_prob * advantage

        critic_loss = nn.functional.mse_loss(
            values,
            returns
        )

        loss = actor_loss + 0.5 * critic_loss

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        rewards_history.append(sum(rewards))

    print(
        "A3C Worker",
        worker_id,
        "Average Reward:",
        round(np.mean(rewards_history), 2)
    )

    return np.mean(rewards_history)


if __name__ == "__main__":

    print("Training A2C")

    a2c_model, a2c_rewards = train_a2c(500)

    print("\nTraining A3C")

    workers = 4

    with mp.Pool(workers) as pool:

        results = pool.starmap(
            train_worker,
            [(i, 125) for i in range(workers)]
        )

    print("\nA2C Average Reward:",
          round(np.mean(a2c_rewards), 2))

    print(
        "A3C Average Reward:",
        round(np.mean(results), 2)
    )
