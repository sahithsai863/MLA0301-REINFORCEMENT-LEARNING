states = ["A", "B", "C", "D", "E"]

routes = {
    "A": {"B": 4, "C": 2},
    "B": {"C": 1, "D": 5},
    "C": {"B": 1, "D": 8, "E": 10},
    "D": {"E": 2},
    "E": {}
}

destination = "E"

cost = {state: float("inf") for state in states}
cost[destination] = 0

for _ in range(len(states) - 1):
    for state in states:
        for next_state, travel_cost in routes[state].items():
            new_cost = travel_cost + cost[next_state]

            if new_cost < cost[state]:
                cost[state] = new_cost

state = "A"
path = [state]

while state != destination:
    next_state = min(
        routes[state],
        key=lambda x: routes[state][x] + cost[x]
    )
    state = next_state
    path.append(state)

print("Minimum Travel Costs:")
for state in states:
    print(state, ":", cost[state])

print("\nOptimal Path:")
print(" -> ".join(path))

print("\nMinimum Travel Cost:")
print(cost["A"])
