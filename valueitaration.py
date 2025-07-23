import numpy as np

# Define MDP parameters
num_states = 2  # 0: Online, 1: Offline
num_actions = 2  # 0: Notify, 1: DoNothing
gamma = 1.0
T = 10  # Finite horizon
theta = 0.001 # Convergence threshold (optional for finite horizon)
 
# Transition probabilities: P[s][a][s']
P = np.zeros((num_states, num_actions, num_states))
P[0,0,:] = [0.9, 0.1]  # Online, Notify -> Online, Offline
P[0,1,:] = [0.7, 0.3]  # Online, DoNothing -> Online, Offline
P[1,0,:] = [0.4, 0.6]  # Offline, Notify -> Online, Offline
P[1,1,:] = [0.2, 0.8]  # Offline, DoNothing -> Online, Offline

# Rewards: r[s][a][s'] (sparse)
r = np.zeros((num_states, num_actions, num_states))
r[0,0,0] = 1.0  # Online, Notify, to Online
r[0,1,1] = -1.0  # Online, DoNothing, to Offline

# Compute expected reward R[s,a]
R = np.sum(P * r, axis=2)

# Finite-horizon value iteration (backward dynamic programming)
V = [np.zeros(num_states) for _ in range(T + 1)]
Q = [np.zeros((num_states, num_actions)) for _ in range(T + 1)]

# Backward pass
for t in range(T - 1, -1, -1):
    for s in range(num_states):
        for a in range(num_actions):
            Q[t][s, a] = R[s, a] + gamma * np.sum(P[s, a, :] * V[t + 1])
        V[t][s] = np.max(Q[t][s])
    print(f"t={t}: V = {np.round(V[t], 3)}")  # Debug: print value function at each t

# Extract time-dependent policy
policy = [np.zeros(num_states, dtype=int) for _ in range(T)]
for t in range(T):
    for s in range(num_states):
        policy[t][s] = np.argmax(Q[t][s])

# Print results
print("\nOptimal Value Function at t=0:")
print("V_0(Online):", round(V[0][0], 2))
print("V_0(Offline):", round(V[0][1], 2))
print("\nOptimal Policy at each t (0: Notify, 1: DoNothing):")
for t in range(T):
    print(f"t={t}: Policy(Online): {policy[t][0]}, Policy(Offline): {policy[t][1]}")