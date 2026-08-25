states = ["A", "B", "C", "D", "E", "F"]

routes = {
    "A": {"B": 4, "C": 2},
    "B": {"D": 5, "E": 10},
    "C": {"B": 1, "D": 8, "E": 7},
    "D": {"E": 2, "F": 6},
    "E": {"F": 3},
    "F": {}
}

destination = "F"

V = {state: float("inf") for state in states}
V[destination] = 0

policy = {}

for _ in range(len(states)):
    for state in states:
        if state == destination:
            continue

        best_cost = float("inf")
        best_action = None

        for next_state, cost in routes[state].items():
            total_cost = cost + V[next_state]

            if total_cost < best_cost:
                best_cost = total_cost
                best_action = next_state

        V[state] = best_cost
        policy[state] = best_action

state = "A"
optimal_route = [state]

while state != destination:
    state = policy[state]
    optimal_route.append(state)

print("Optimal State Values:")

for state in states:
    print(state, ":", V[state])

print("\nOptimal Driving Policy:")

for state in policy:
    print(state, "->", policy[state])

print("\nOptimal Taxi Route:")
print(" -> ".join(optimal_route))

print("\nMinimum Travel Cost:")
print(V["A"])
