import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MDP:
    def __init__(self, num_states, num_actions, reward_matrix, init_dist, pi_b_matrix, pi_e_matrix):
        self.num_states = num_states
        self.num_actions = num_actions
        self.states = [f"s{i}" for i in range(num_states)]
        self.actions = [f"a{j}" for j in range(num_actions)]
        self.reward_matrix = reward_matrix
        self.init_dist = init_dist
        self.pi_b_matrix = pi_b_matrix
        self.pi_e_matrix = pi_e_matrix
        self.q_matrix = self.reward_matrix.copy()
        self.v = self.compute_v(self.pi_e_matrix, self.q_matrix)
        self.p_b = self.init_dist[:, None] * self.pi_b_matrix
        self.p_e = self.init_dist[:, None] * self.pi_e_matrix
        self.mu = self.p_e / self.p_b
        self.rho_true = np.sum(self.init_dist * np.sum(self.pi_e_matrix * self.reward_matrix, axis=1))

    def compute_v(self, pi_e, q):
        return np.sum(pi_e * q, axis=1)

    def sample_trajectory(self, n=1000):
        data = []
        for _ in range(n):
            s_idx = np.random.choice(self.num_states, p=self.init_dist)
            a_idx = np.random.choice(self.num_actions, p=self.pi_b_matrix[s_idx])
            r = self.reward_matrix[s_idx, a_idx]
            data.append((s_idx, a_idx, r))
        return data

    def estimate_values(self, data):
        v_naive = np.mean([self.v[s] for (s, a, r) in data])
        is_weights = [self.mu[s, a] for (s, a, r) in data]
        is_values = [w * r for w, (s, a, r) in zip(is_weights, data)]
        v_is = np.mean(is_values)
        eff_values = [self.mu[s, a] * (r - self.q_matrix[s, a]) + self.v[s] for (s, a, r) in data]
        v_eff = np.mean(eff_values)
        return v_naive, v_is, v_eff

    def visualize_trajectories(self, data, num=10):
        print("\nSample Trajectories:")
        for i, (s, a, r) in enumerate(data[:num]):
            print(f"Trajectory {i+1}: State={self.states[s]}, Action={self.actions[a]}, Reward={r:.2f}")

    def plot_rewards_distribution(self, data):
        rewards = [r for (_, _, r) in data]
        plt.hist(rewards, bins=20, edgecolor='black')
        plt.title("Reward Distribution from Simulated Trajectories")
        plt.xlabel("Reward")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

# Define MDP parameters manually
num_states = 2
num_actions = 2
reward_matrix = np.array([[1.0, 2.0], [0.0, 3.0]])
init_dist = np.array([0.6, 0.4])
pi_b_matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
pi_e_matrix = np.array([[0.5, 0.5], [0.2, 0.8]])

# Run the simulation
np.random.seed(42)
mdp = MDP(num_states, num_actions, reward_matrix, init_dist, pi_b_matrix, pi_e_matrix)
data = mdp.sample_trajectory(n=1000)
v_naive, v_is, v_eff = mdp.estimate_values(data)

print("True policy value (rho^pi_e):", round(mdp.rho_true, 3))
print("Naive plug-in estimator:", round(v_naive, 3))
print("Importance Sampling estimator:", round(v_is, 3))
print("Efficient Influence Function estimator:", round(v_eff, 3))

# Visualize results
mdp.visualize_trajectories(data)
mdp.plot_rewards_distribution(data)




# Run alternative MDP
print("\nAlternative MDP (3 states):")
mdp_alt = MDP(num_states, num_actions, T, s0, P, R, pi_b, pi_e)
run_simulation(mdp_alt, n_trajectories=1000, seed=10)


# Alternative configuration (3 states, different policies)
num_states = 3
num_actions = 2
T = 2
s0 = 0

# Example transition matrix (ring structure)
P = np.zeros((num_states, num_actions, num_states))
for s in range(num_states):
    for a in range(num_actions):
        s_next = (s + a + 1) % num_states
        P[s, a, s_next] = 1.0

# Reward matrix
R = np.ones((num_states, num_actions)) * 0.5
R[0, 1] = 1.0  # Bonus reward

# Policies
pi_b = np.ones((num_states, num_actions)) / num_actions  # Uniform
pi_e = np.zeros((num_states, num_actions))
pi_e[:, 1] = 1.0  # Always action 1'