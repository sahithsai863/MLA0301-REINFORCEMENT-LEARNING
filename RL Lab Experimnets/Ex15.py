import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

STATE_SIZE = 6
ACTION_SIZE = 5
MAX_STEPS = 100


class HumanoidEnv:

    def __init__(self):
        self.reset()

    def reset(self):
        self.position = 0.0
        self.velocity = 0.0
        self.angle = 0.0
        self.angular_velocity = 0.0
        self.leg_position = 0.0
        self.steps = 0
        return self.state()

    def state(self):
        return np.array([
            self.position,
            self.velocity,
            self.angle,
            self.angular_velocity,
            self.leg_position,
            self.steps / MAX_STEPS
        ], dtype=np.float32)

    def step(self, action):

        self.steps += 1

        if action == 0:
            self.velocity += 0.08
            self.leg_position += 0.05

        elif action == 1:
            self.velocity -= 0.08
            self.leg_position -= 0.05

        elif action == 2:
            self.velocity += 0.15
            self.leg_position += 0.02

        elif action == 3:
            self.velocity -= 0.10
            self.leg_position -= 0.02

        elif action == 4:
            self.velocity *= 0.8
            self.angular_velocity *= 0.5

        self.position += self.velocity

        self.angular_velocity += (
            np.random.normal(0, 0.02)
            + self.velocity * 0.01
        )

        self.angle += self.angular_velocity

        balance_reward = 5 - abs(self.angle) * 10

        movement_reward = max(
            0,
            self.velocity * 5
        )

        reward = (
            balance_reward
            + movement_reward
            + 1
        )

        done = False

        if abs(self.angle) > 1.0:
            reward -= 50
            done = True

        if self.steps >= MAX_STEPS:
            done = True

        return self.state(), reward, done


class ActorCritic(nn.Module):

    def __init__(self):

        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(STATE_SIZE, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )

        self.actor = nn.Linear(
            128,
            ACTION_SIZE
        )

        self.critic = nn.Linear(
            128,
            1
        )

    def forward(self, state):

        x = self.shared(state)

        policy = torch.softmax(
            self.actor(x),
            dim=-1
        )

        value = self.critic(x)

        return policy, value


def collect_data(model, env):

    states = []
    actions = []
    rewards = []
    log_probs = []
    values = []

    state = env.reset()

    for _ in range(MAX_STEPS):

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        )

        policy, value = model(
            state_tensor
        )

        distribution = Categorical(
            policy
        )

        action = distribution.sample()

        next_state, reward, done = env.step(
            action.item()
        )

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(
            distribution.log_prob(action)
        )
        values.append(value.squeeze())

        state = next_state

        if done:
            break

    returns = []

    G = 0

    for reward in reversed(rewards):

        G = reward + 0.99 * G

        returns.insert(0, G)

    return (
        torch.tensor(
            np.array(states),
            dtype=torch.float32
        ),
        torch.stack(actions),
        torch.tensor(
            returns,
            dtype=torch.float32
        ),
        torch.stack(log_probs).detach(),
        torch.stack(values).detach()
    )


def train_ppo(episodes=500):

    model = ActorCritic()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.0003
    )

    clip = 0.2

    reward_history = []

    for episode in range(episodes):

        env = HumanoidEnv()

        states, actions, returns, old_log_probs, old_values = \
            collect_data(model, env)

        for _ in range(5):

            policies, values = model(states)

            distribution = Categorical(
                policies
            )

            new_log_probs = distribution.log_prob(
                actions
            )

            advantages = (
                returns - values.squeeze()
            ).detach()

            ratio = torch.exp(
                new_log_probs - old_log_probs
            )

            clipped_ratio = torch.clamp(
                ratio,
                1 - clip,
                1 + clip
            )

            actor_loss = -torch.min(
                ratio * advantages,
                clipped_ratio * advantages
            ).mean()

            critic_loss = (
                returns - values.squeeze()
            ).pow(2).mean()

            loss = (
                actor_loss
                + 0.5 * critic_loss
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        total_reward = returns[0].item()

        reward_history.append(
            total_reward
        )

        if (episode + 1) % 50 == 0:

            print(
                "PPO Episode:",
                episode + 1,
                "Reward:",
                round(total_reward, 2)
            )

    return model, reward_history


def train_trpo(episodes=500):

    model = ActorCritic()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.0001
    )

    reward_history = []

    for episode in range(episodes):

        env = HumanoidEnv()

        states, actions, returns, old_log_probs, old_values = \
            collect_data(model, env)

        policies, values = model(states)

        distribution = Categorical(
            policies
        )

        new_log_probs = distribution.log_prob(
            actions
        )

        advantages = (
            returns - values.squeeze()
        ).detach()

        ratio = torch.exp(
            new_log_probs - old_log_probs
        )

        surrogate = (
            ratio * advantages
        ).mean()

        kl = torch.distributions.kl_divergence(
            Categorical(
                torch.exp(old_log_probs).detach()
            ),
            distribution
        ).mean()

        loss = (
            -surrogate
            + 0.5 * (
                returns - values.squeeze()
            ).pow(2).mean()
        )

        if kl.item() < 0.01:

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        total_reward = returns[0].item()

        reward_history.append(
            total_reward
        )

        if (episode + 1) % 50 == 0:

            print(
                "TRPO Episode:",
                episode + 1,
                "Reward:",
                round(total_reward, 2)
            )

    return model, reward_history


print("Training PPO")

ppo_model, ppo_rewards = train_ppo()

print("\nTraining TRPO")

trpo_model, trpo_rewards = train_trpo()

print("\nTraining Completed")

print(
    "\nPPO Average Reward:",
    round(np.mean(ppo_rewards[-50:]), 2)
)

print(
    "TRPO Average Reward:",
    round(np.mean(trpo_rewards[-50:]), 2)
)
