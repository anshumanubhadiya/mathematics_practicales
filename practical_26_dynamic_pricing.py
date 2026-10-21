"""Practical 26: AI-based dynamic pricing using Q-learning."""
import numpy as np

np.random.seed(42)  # reproducible output ke liye

# Actions (Prices) & Demand States (0 = Low, 1 = Medium, 2 = High)
prices = np.array([4, 5, 6, 7, 8])
Q = np.zeros((3, len(prices)))  # Q-table (States x Actions)

# Q-learning parameters
alpha = 0.1    # Learning rate
gamma = 0.9    # Discount factor
epsilon = 0.2  # Exploration rate
n_episodes = 1000

# Simulated demand matrix (quantity sold for each state and price)
demand_matrix = np.array([
    [4, 9, 10, 11, 12],    # Low demand
    [5, 9, 11, 13, 14],    # Medium demand
    [6, 10, 14, 15, 16]    # High demand
])

for episode in range(n_episodes):
    state = np.random.choice(3)

    # Epsilon-greedy action selection
    if np.random.rand() < epsilon:
        action = np.random.randint(len(prices))  # Explore
    else:
        action = np.argmax(Q[state])  # Exploit

    quantity_sold = demand_matrix[state, action]
    reward = prices[action] * quantity_sold

    # Q-value update
    Q[state, action] += alpha * (reward + gamma * np.max(Q[state]) - Q[state, action])

print("Learned Q-table (State x Price):\n", Q)