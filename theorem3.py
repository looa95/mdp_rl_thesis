import numpy as np

class MDP:
    def __init__(self, num_states, num_actions, T, s0, P, R, pi_b, pi_e):
        """
        Initialize MDP with the following parameters:
        - num_states: Number of states
        - num_actions: Number of actions
        - T: Horizon (number of time steps)
        - s0: Initial state index
        - P: Transition matrix, shape (num_states, num_actions, num_states)
        - R: Reward matrix, shape (num_states, num_actions)
        - pi_b: Behavior policy matrix, shape (num_states, num_actions)
        - pi_e: Target policy matrix, shape (num_states, num_actions)
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.T = T
        self.s0 = s0
        self.P = P  # P[s, a, s'] = P(s' | s, a)
        self.R = R  # R[s, a] = reward
        self.pi_b = pi_b  # pi_b[s, a] = P(a | s)
        self.pi_e = pi_e  # pi_e[s, a] = P(a | s)
        
        # Validate inputs
        assert P.shape == (num_states, num_actions, num_states)
        assert R.shape == (num_states, num_actions)
        assert pi_b.shape == (num_states, num_actions)
        assert pi_e.shape == (num_states, num_actions)
        assert np.allclose(pi_b.sum(axis=1), 1.0), "pi_b rows must sum to 1"
        assert np.allclose(pi_e.sum(axis=1), 1.0), "pi_e rows must sum to 1"
        assert np.allclose(P.sum(axis=2), 1.0), "P[:, a, :] must sum to 1"

    def compute_values(self):
        """
        Compute V_t (value function) and Q_t(state, value function) matrices for all t using backward induction.
        Returns:
        - Q: List of Q_t matrices, shape (T+1, num_states, num_actions)
        - V: List of V_t arrays, shape (T+1, num_states)
        """
        Q = [np.zeros((self.num_states, self.num_actions)) for _ in range(self.T + 1)]
        V = [np.zeros(self.num_states) for _ in range(self.T + 1)]
        
        # Terminal step (t=T): Q_T(s, a) = R(s, a)
        Q[self.T] = self.R.copy()
        V[self.T] = np.sum(self.pi_e * Q[self.T], axis=1)  # V_T(s) = sum_a pi_e(s, a) Q_T(s, a)
        
        # Backward induction
        for t in range(self.T - 1, -1, -1):
            for s in range(self.num_states):
                for a in range(self.num_actions):
                    # Q_t(s, a) = R(s, a) + sum_s' P(s, a, s') V_{t+1}(s')
                    Q[t][s, a] = self.R[s, a] + np.sum(self.P[s, a] * V[t + 1])
                # V_t(s) = sum_a pi_e(s, a) Q_t(s, a)
                V[t][s] = np.sum(self.pi_e[s] * Q[t][s])
        
        return Q, V

    def compute_cum_reward(self, policy):
        """
        Average cummulative reward of the policy pi(s, a) over time.
        Compute p_pi(t, s, a) for all t using forward recursion.
        Returns:
        - p: Array of shape (T+1, num_states, num_actions)
        """
        p = np.zeros((self.T + 1, self.num_states, self.num_actions))
        
        # Initial distribution: s0 fixed
        p[0, self.s0, :] = policy[self.s0]
        
        # Forward recursion
        for t in range(1, self.T + 1):
            for s in range(self.num_states):
                for a in range(self.num_actions):
                    # p(t, s, a) = sum_s',a' p(t-1, s', a') P(s', a', s) pi(a | s)
                    for sp in range(self.num_states):
                        for ap in range(self.num_actions):
                            p[t, s, a] += p[t-1, sp, ap] * self.P[sp, ap, s] * policy[s, a]
        
        return p

    def compute_mu(self):
        """
        Compute mu_t(s, a) = p_pi^e(t, s, a) / p_pi^b(t, s, a).
        Returns:
        - mu: List of mu_t matrices, shape (T+1, num_states, num_actions)
        """
        p_pi_e = self.compute_cum_reward(self.pi_e)
        p_pi_b = self.compute_cum_reward(self.pi_b)
        
        mu = []
        for t in range(self.T + 1):
            mu_t = np.zeros((self.num_states, self.num_actions))
            for s in range(self.num_states):
                for a in range(self.num_actions):
                    if p_pi_b[t, s, a] > 0:
                        mu_t[s, a] = p_pi_e[t, s, a] / p_pi_b[t, s, a]
            mu.append(mu_t)
        
        return mu

    def simulate_trajectory(self):
        """Simulate a trajectory under pi^b."""
        trajectory = []
        s = self.s0
        for t in range(self.T + 1):
            a = np.random.choice(self.num_actions, p=self.pi_b[s])
            r = self.R[s, a]
            trajectory.append((s, a, r))
            # Sample next state
            if t < self.T:
                s = np.random.choice(self.num_states, p=self.P[s, a])
        return trajectory

    def compute_eif(self, trajectory, rho, Q, V, mu):
        """Compute the EIF for a trajectory."""
        phi = -rho
        terms = 0.0
        
        for t in range(self.T + 1):
            s_t, a_t, r_t = trajectory[t]
            # mu_t * (r_t - Q_t)
            mu_term = mu[t][s_t, a_t] * (r_t - Q[t][s_t, a_t])
            terms += mu_term
            # mu_{t-1} * V_t
            if t == 0:
                mu_prev_v = V[0][s_t]  # mu_{-1} = 1
            else:
                s_prev, a_prev, _ = trajectory[t-1]
                mu_prev_v = mu[t-1][s_prev, a_prev] * V[t][s_t]
            terms += mu_prev_v
        
        phi += terms
        return phi, terms

def run_simulation(mdp, n_trajectories, seed):
    """Run simulation and estimate rho^{pi^e}."""
    # Compute necessary functions
    Q, V = mdp.compute_values()
    mu = mdp.compute_mu()
    rho_true = V[0][mdp.s0]  # True value starting from s0
    
    # Simulate trajectories
    eif_values = []
    terms_values = []
    np.random.seed(seed)
    for _ in range(n_trajectories):
        traj = mdp.simulate_trajectory()
        phi, terms = mdp.compute_eif(traj, rho_true, Q, V, mu)
        eif_values.append(phi)
        terms_values.append(terms)
    
    # Compute estimates
    average_eif = np.mean(eif_values)
    average_terms = np.mean(terms_values)
    hat_rho = average_terms
    
    # Output results
    print(f"True rho^{{pi^e}}: {rho_true:.4f}")
    print(f"Average EIF: {average_eif:.4f} ")
    print(f"Estimated rho^{{pi^e}}: {hat_rho:.4f}")
    
    # Example trajectories
    print("\nExample Trajectories and EIFs:")
    for i in range(3):
        traj = mdp.simulate_trajectory()
        phi, terms = mdp.compute_eif(traj, rho_true, Q, V, mu)
        print(f"Trajectory {i+1}: {traj}")
        print(f"  EIF: {phi:.4f}, Terms: {terms:.4f}")

# Example configuration (same as previous example)
num_states = 2
num_actions = 2
T = 1
s0 = 0

# Transition matrix // this transition matrix is just an indication of the possible state change after taking some action. 
P = np.zeros((num_states, num_actions, num_states))
P[0, 0, 0] = 1.0  # s=0, a=0 -> s'=0
P[0, 1, 1] = 1.0  # s=0, a=1 -> s'=1
P[1, :, 1] = 1.0   # s=1, any a -> s'=1

# Reward matrix
R = np.zeros((num_states, num_actions))
R[0, 1] = 1.0  # s=0, a=1 -> r=1

# Policies
pi_b = np.array([
    [0.5, 0.5],
    [1, 0]
])
pi_e = np.array([
    [0, 1],
    [0, 1]
])

# Create and run MDP
mdp = MDP(num_states, num_actions, T, s0, P, R, pi_b, pi_e)
run_simulation(mdp, n_trajectories=2, seed=10)


