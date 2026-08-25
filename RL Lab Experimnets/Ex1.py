import numpy as np

# ---------------------------------
# Simplified Chess MDP
# ---------------------------------

# States
states = ["Start", "Attack", "Defend", "Win", "Lose"]

# Possible Actions
actions = {
    "Start": ["Attack", "Defend"],
    "Attack": ["Capture", "Retreat"],
    "Defend": ["Counter", "Wait"],
    "Win": [],
    "Lose": []
}

# ---------------------------------
# Transition Probabilities
# ---------------------------------

transition = {

    ("Start", "Attack"): [("Attack", 1.0)],
    ("Start", "Defend"): [("Defend", 1.0)],

    # Capture has higher probability of winning
    ("Attack", "Capture"): [("Win", 0.9), ("Lose", 0.1)],

    # Retreat always goes back to Defend
    ("Attack", "Retreat"): [("Defend", 1.0)],

    ("Defend", "Counter"): [("Win", 0.8), ("Attack", 0.2)],

    ("Defend", "Wait"): [("Lose", 0.4), ("Defend", 0.6)]
}

# ---------------------------------
# Rewards
# ---------------------------------

reward = {
    "Start": 0,
    "Attack": 20,
    "Defend": -5,
    "Win": 100,
    "Lose": -100
}

# Discount Factor
gamma = 0.9

# ---------------------------------
# Initialize Value Function
# ---------------------------------

V = {state: 0 for state in states}

# ---------------------------------
# Value Iteration
# ---------------------------------

iterations = 50

for i in range(iterations):

    new_V = V.copy()

    for state in states:

        if state in ["Win", "Lose"]:
            continue

        action_values = []

        for action in actions[state]:

            value = 0

            for next_state, prob in transition[(state, action)]:

                r = reward[next_state]
                value += prob * (r + gamma * V[next_state])

            action_values.append(value)

        new_V[state] = max(action_values)

    V = new_V

# ---------------------------------
# Extract Optimal Policy
# ---------------------------------

policy = {}

for state in states:

    if state in ["Win", "Lose"]:
        policy[state] = "-"
        continue

    best_action = None
    best_value = -float("inf")

    for action in actions[state]:

        value = 0

        for next_state, prob in transition[(state, action)]:

            r = reward[next_state]
            value += prob * (r + gamma * V[next_state])

        if value > best_value:
            best_value = value
            best_action = action

    policy[state] = best_action

# ---------------------------------
# Display Results
# ---------------------------------

print("=" * 40)
print("Simplified Chess MDP using Value Iteration")
print("=" * 40)


print("\nState Values:\n")

for state in states:
    print(f"{state:<8} : {V[state]:.2f}")

print("\nOptimal Policy:\n")

for state in states:
    print(f"{state:<8} --> {policy[state]}")