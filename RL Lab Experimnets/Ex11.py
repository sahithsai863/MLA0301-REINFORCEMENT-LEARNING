import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STATE_SIZE = 5
ACTION_SIZE = 2

EPISODES = 150
STEPS = 100
BATCH_SIZE = 64
GAMMA = 0.95
LR = 0.001
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.985
TARGET_UPDATE = 10


class TrafficEnvironment:
    def __init__(self):
        self.reset()

    def reset(self):
        self.queues = np.random.randint(0, 10, 4).astype(float)
        self.signal = random.randint(0, 1)
        self.waiting_time = 0
        self.total_passed = 0
        return self.get_state()

    def get_state(self):
        total_queue = np.sum(self.queues)

        state = np.array([
            self.queues[0] / 20.0,
            self.queues[1] / 20.0,
            self.queues[2] / 20.0,
            self.queues[3] / 20.0,
            self.signal
        ], dtype=np.float32)

        return state

    def step(self, action):
        if action == 1:
            self.signal = 1 - self.signal

        arrivals = np.random.poisson(2, 4)

        self.queues += arrivals

        if self.signal == 0:
            green_lanes = [0, 1]
        else:
            green_lanes = [2, 3]

        passed = 0

        for lane in green_lanes:
            vehicles = min(self.queues[lane], random.randint(2, 5))
            self.queues[lane] -= vehicles
            passed += vehicles

        self.queues = np.maximum(self.queues, 0)

        current_waiting = np.sum(self.queues)
        self.waiting_time += current_waiting
        self.total_passed += passed

        reward = -current_waiting + passed * 0.5

        next_state = self.get_state()

        return next_state, reward, False, {
            "waiting_time": current_waiting,
            "queue_length": current_waiting,
            "vehicles_passed": passed
        }


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(
            (state, action, reward, next_state, done)
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta

        self.buffer = []
        self.priorities = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):

        max_priority = max(self.priorities) if self.priorities else 1.0

        experience = (
            state,
            action,
            reward,
            next_state,
            done
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = max_priority

            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):

        priorities = np.array(self.priorities, dtype=np.float32)

        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(
            len(self.buffer),
            batch_size,
            p=probabilities
        )

        samples = [self.buffer[i] for i in indices]

        weights = (
            len(self.buffer) * probabilities[indices]
        ) ** (-self.beta)

        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            weights
        )

    def update_priorities(self, indices, errors):

        for index, error in zip(indices, errors):
            self.priorities[index] = abs(error) + 1e-6

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(STATE_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_SIZE)
        )

    def forward(self, x):
        return self.network(x)


class DuelingDQN(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(STATE_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_SIZE)
        )

    def forward(self, x):

        features = self.feature(x)

        value = self.value_stream(features)

        advantage = self.advantage_stream(features)

        q_values = value + (
            advantage - advantage.mean(dim=1, keepdim=True)
        )

        return q_values


def select_action(model, state, epsilon):

    if random.random() < epsilon:
        return random.randrange(ACTION_SIZE)

    state_tensor = torch.FloatTensor(
        state
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        q_values = model(state_tensor)

    return q_values.argmax().item()


def train_dqn(
    model,
    target_model,
    optimizer,
    memory,
    double=False,
    prioritized=False
):

    if len(memory) < BATCH_SIZE:
        return

    if prioritized:

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            indices,
            weights
        ) = memory.sample(BATCH_SIZE)

        weights = torch.FloatTensor(weights).to(DEVICE)

    else:

        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = memory.sample(BATCH_SIZE)

        weights = torch.ones(BATCH_SIZE).to(DEVICE)

    states = torch.FloatTensor(states).to(DEVICE)
    actions = torch.LongTensor(actions).to(DEVICE)
    rewards = torch.FloatTensor(rewards).to(DEVICE)
    next_states = torch.FloatTensor(next_states).to(DEVICE)
    dones = torch.FloatTensor(dones).to(DEVICE)

    current_q = model(states).gather(
        1,
        actions.unsqueeze(1)
    ).squeeze(1)

    with torch.no_grad():

        if double:

            next_actions = model(
                next_states
            ).argmax(1)

            next_q = target_model(
                next_states
            ).gather(
                1,
                next_actions.unsqueeze(1)
            ).squeeze(1)

        else:

            next_q = target_model(
                next_states
            ).max(1)[0]

        target_q = rewards + (
            GAMMA * next_q * (1 - dones)
        )

    td_errors = target_q - current_q

    loss = (
        weights * td_errors.pow(2)
    ).mean()

    optimizer.zero_grad()

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0
    )

    optimizer.step()

    if prioritized:
        errors = td_errors.detach().cpu().numpy()
        memory.update_priorities(
            indices,
            errors
        )


def train_algorithm(name):

    env = TrafficEnvironment()

    if name == "Dueling DQN":
        model = DuelingDQN().to(DEVICE)
        target_model = DuelingDQN().to(DEVICE)
    else:
        model = DQN().to(DEVICE)
        target_model = DQN().to(DEVICE)

    target_model.load_state_dict(
        model.state_dict()
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=LR
    )

    if name == "DQN + PER":
        memory = PrioritizedReplayBuffer()
        prioritized = True
    else:
        memory = ReplayBuffer()
        prioritized = False

    double = name in [
        "DDQN",
        "Dueling DQN",
        "DQN + PER"
    ]

    rewards_history = []
    waiting_history = []
    queue_history = []
    throughput_history = []

    epsilon = EPSILON_START

    for episode in range(EPISODES):

        state = env.reset()

        episode_reward = 0
        episode_waiting = 0
        episode_queue = 0
        episode_passed = 0

        for step in range(STEPS):

            action = select_action(
                model,
                state,
                epsilon
            )

            next_state, reward, done, info = env.step(
                action
            )

            memory.push(
                state,
                action,
                reward,
                next_state,
                done
            )

            train_dqn(
                model,
                target_model,
                optimizer,
                memory,
                double=double,
                prioritized=prioritized
            )

            state = next_state

            episode_reward += reward
            episode_waiting += info["waiting_time"]
            episode_queue += info["queue_length"]
            episode_passed += info["vehicles_passed"]

        epsilon = max(
            EPSILON_MIN,
            epsilon * EPSILON_DECAY
        )

        if episode % TARGET_UPDATE == 0:
            target_model.load_state_dict(
                model.state_dict()
            )

        rewards_history.append(
            episode_reward
        )

        waiting_history.append(
            episode_waiting / STEPS
        )

        queue_history.append(
            episode_queue / STEPS
        )

        throughput_history.append(
            episode_passed
        )

        if (episode + 1) % 10 == 0:

            print(
                f"{name} | "
                f"Episode {episode + 1}/{EPISODES} | "
                f"Reward: {episode_reward:.2f} | "
                f"Waiting: "
                f"{episode_waiting / STEPS:.2f} | "
                f"Queue: "
                f"{episode_queue / STEPS:.2f} | "
                f"Passed: "
                f"{episode_passed}"
            )

    return {
        "reward": rewards_history,
        "waiting": waiting_history,
        "queue": queue_history,
        "throughput": throughput_history
    }


results = {}

algorithms = [
    "DQN",
    "DDQN",
    "Dueling DQN",
    "DQN + PER"
]

for algorithm in algorithms:

    print("\n==============================")
    print("Training:", algorithm)
    print("==============================")

    results[algorithm] = train_algorithm(
        algorithm
    )


plt.figure(figsize=(10, 6))

for algorithm in algorithms:

    plt.plot(
        results[algorithm]["reward"],
        label=algorithm
    )

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward Comparison")
plt.legend()
plt.grid()

plt.show()


plt.figure(figsize=(10, 6))

for algorithm in algorithms:

    plt.plot(
        results[algorithm]["waiting"],
        label=algorithm
    )

plt.xlabel("Episode")
plt.ylabel("Average Waiting Time")
plt.title("Waiting Time Comparison")
plt.legend()
plt.grid()

plt.show()


plt.figure(figsize=(10, 6))

for algorithm in algorithms:

    plt.plot(
        results[algorithm]["queue"],
        label=algorithm
    )

plt.xlabel("Episode")
plt.ylabel("Average Queue Length")
plt.title("Queue Length Comparison")
plt.legend()
plt.grid()

plt.show()


plt.figure(figsize=(10, 6))

for algorithm in algorithms:

    plt.plot(
        results[algorithm]["throughput"],
        label=algorithm
    )

plt.xlabel("Episode")
plt.ylabel("Vehicles Passed")
plt.title("Traffic Throughput Comparison")
plt.legend()
plt.grid()

plt.show()


print("\n========== FINAL COMPARISON ==========")

for algorithm in algorithms:

    avg_waiting = np.mean(
        results[algorithm]["waiting"][-20:]
    )

    avg_queue = np.mean(
        results[algorithm]["queue"][-20:]
    )

    avg_throughput = np.mean(
        results[algorithm]["throughput"][-20:]
    )

    avg_reward = np.mean(
        results[algorithm]["reward"][-20:]
    )

    print(
        f"\n{algorithm}"
    )

    print(
        f"Average Waiting Time : {avg_waiting:.2f}"
    )

    print(
        f"Average Queue Length : {avg_queue:.2f}"
    )

    print(
        f"Average Throughput   : {avg_throughput:.2f}"
    )

    print(
        f"Average Reward       : {avg_reward:.2f}"
    )
