import random

ads = ["Ad_A", "Ad_B", "Ad_C", "Ad_D"]
true_rewards = [0.20, 0.50, 0.80, 0.35]

epsilon = 0.1
episodes = 1000

counts = [0] * len(ads)
estimated_rewards = [0.0] * len(ads)

total_reward = 0

for _ in range(episodes):
    if random.random() < epsilon:
        ad = random.randint(0, len(ads) - 1)
    else:
        ad = estimated_rewards.index(max(estimated_rewards))

    reward = 1 if random.random() < true_rewards[ad] else 0

    counts[ad] += 1
    estimated_rewards[ad] += (
        reward - estimated_rewards[ad]
    ) / counts[ad]

    total_reward += reward

best_ad = estimated_rewards.index(max(estimated_rewards))
engagement_rate = total_reward / episodes

print("Advertisement Statistics:")

for i in range(len(ads)):
    print(
        ads[i],
        "Selections:", counts[i],
        "Estimated Reward:", round(estimated_rewards[i], 3)
    )

print("\nTotal Reward:", total_reward)
print("Overall Engagement Rate:", round(engagement_rate, 3))
print("Best Advertisement:", ads[best_ad])
