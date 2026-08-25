states = [
    "Start",
    "Aisle_1",
    "Aisle_2",
    "Storage",
    "Pickup",
    "Delivery",
    "Obstacle"
]

actions = {
    "Start": ["Aisle_1"],
    "Aisle_1": ["Aisle_2", "Obstacle"],
    "Aisle_2": ["Storage", "Obstacle"],
    "Storage": ["Pickup"],
    "Pickup": ["Delivery"],
    "Delivery": []
}

rewards = {
    "Start": 0,
    "Aisle_1": 5,
    "Aisle_2": 10,
    "Storage": 20,
    "Pickup": 50,
    "Delivery": 100,
    "Obstacle": -50
}

gamma = 0.9
values = {state: 0 for state in states}

for _ in range(100):
    new_values = values.copy()

    for state in actions:
        if not actions[state]:
            new_values[state] = rewards[state]
            continue

        best_value = float("-inf")

        for next_state in actions[state]:
            value = rewards[next_state] + gamma * values[next_state]

            if value > best_value:
                best_value = value

        new_values[state] = best_value

    values = new_values

state = "Start"
path = [state]

while actions[state]:
    best_action = None
    best_value = float("-inf")

    for next_state in actions[state]:
        value = rewards[next_state] + gamma * values[next_state]

        if value > best_value:
            best_value = value
            best_action = next_state

    state = best_action
    path.append(state)

print("State Values:")
for state in states:
    print(state, ":", round(values[state], 2))

print("\nOptimal Robot Path:")
print(" -> ".join(path))

print("\nReward Obtained:")
print(rewards[path[-1]])

print("\nResult:")
if path[-1] == "Delivery":
    print("Robot successfully completed the delivery.")
else:
    print("Robot failed to complete the delivery.")
