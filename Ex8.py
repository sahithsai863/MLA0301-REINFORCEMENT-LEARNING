import random

states = [
    "Dirty_Left",
    "Dirty_Right",
    "Clean_Left",
    "Clean_Right"
]

actions = [
    "Clean",
    "Move"
]

rewards = {
    ("Dirty_Left", "Clean"): 10,
    ("Dirty_Right", "Clean"): 10,
    ("Dirty_Left", "Move"): -1,
    ("Dirty_Right", "Move"): -1,
    ("Clean_Left", "Move"): -1,
    ("Clean_Right", "Move"): -1
}

gamma = 0.9
episodes = 1000

def next_state(state, action):
    if action == "Clean":
        if state == "Dirty_Left":
            return "Clean_Left"
        elif state == "Dirty_Right":
            return "Clean_Right"
    elif action == "Move":
        if state == "Dirty_Left":
            return "Dirty_Right"
        elif state == "Dirty_Right":
            return "Dirty_Left"
        elif state == "Clean_Left":
            return "Clean_Right"
        elif state == "Clean_Right":
            return "Clean_Left"
    return state

Q = {}
returns = {}

for state in states:
    for action in actions:
        Q[(state, action)] = 0
        returns[(state, action)] = []

def choose_action(state):
    values = [Q.get((state, action), 0) for action in actions]
    max_value = max(values)
    best_actions = [
        actions[i]
        for i in range(len(actions))
        if values[i] == max_value
    ]
    return random.choice(best_actions)

for episode_number in range(episodes):
    state = random.choice(["Dirty_Left", "Dirty_Right"])
    episode = []

    for step in range(10):
        action = choose_action(state)
        reward = rewards.get((state, action), -1)
        new_state = next_state(state, action)

        episode.append((state, action, reward))
        state = new_state

        if state in ["Clean_Left", "Clean_Right"]:
            break

    G = 0
    visited = set()

    for state, action, reward in reversed(episode):
        G = gamma * G + reward

        if (state, action) not in visited:
            visited.add((state, action))
            returns[(state, action)].append(G)

            Q[(state, action)] = (
                sum(returns[(state, action)])
                / len(returns[(state, action)])
            )

policy = {}

for state in states:
    best_action = max(
        actions,
        key=lambda action: Q[(state, action)]
    )
    policy[state] = best_action

print("LEARNED Q-VALUES")

for state in states:
    for action in actions:
        print(
            state,
            "+",
            action,
            "=",
            round(Q[(state, action)], 2)
        )

print()
print("OPTIMAL POLICY")

for state in states:
    print(state, "->", policy[state])

print()
print("TESTING THE AGENT")

state = "Dirty_Left"
path = [state]
total_reward = 0
energy = 0

for step in range(10):
    action = policy[state]

    if action == "Clean":
        reward = 10
        energy += 2
    else:
        reward = rewards.get((state, action), -1)
        energy += 1

    total_reward += reward

    state = next_state(state, action)
    path.append(state)

    print(
        "Step:",
        step + 1,
        "| Action:",
        action,
        "| State:",
        state,
        "| Reward:",
        reward
    )

    if state in ["Clean_Left", "Clean_Right"]:
        break

print()
print("FINAL RESULT")
print("Path:", " -> ".join(path))
print("Total Reward:", total_reward)
print("Energy Used:", energy)
print("Monte Carlo Learning Completed!")
