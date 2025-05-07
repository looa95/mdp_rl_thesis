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
        V[self.T] = np.sum(self.pi_e * Q[self.T], axis=1)
        
        # Backward induction
        for t in range(self.T - 1, -1, -1):
            for s in range(self.num_states):
                for a in range(self.num_actions):
                    Q[t][s, a] = self.R[s, a] + np.sum(self.P[s, a] * V[t + 1])
                V[t][s] = np.sum(self.pi_e[s] * Q[t][s])
        return Q, V

    def compute_cum_reward(self, policy):
        """
        Compute average cumulative reward distribution p_pi(t, s, a) via forward recursion.
        Returns:
        - p: Array of shape (T+1, num_states, num_actions)
        """
        p = np.zeros((self.T + 1, self.num_states, self.num_actions))
        p[0, self.s0, :] = policy[self.s0]
        for t in range(1, self.T + 1):
            for s in range(self.num_states):
                for a in range(self.num_actions):
                    for sp in range(self.num_states):
                        for ap in range(self.num_actions):
                            p[t, s, a] += p[t-1, sp, ap] * self.P[sp, ap, s] * policy[s, a]
        return p

    def compute_mu(self):
        """
        Compute importance weights mu_t(s,a)=p_pi^e/p_pi^b for all t.
        Returns:
        - mu: List of mu_t matrices, shape (T+1, num_states, num_actions)
        """
        p_e = self.compute_cum_reward(self.pi_e)
        p_b = self.compute_cum_reward(self.pi_b)
        mu = []
        for t in range(self.T + 1):
            mu_t = np.zeros((self.num_states, self.num_actions))
            for s in range(self.num_states):
                for a in range(self.num_actions):
                    if p_b[t, s, a] > 0:
                        mu_t[s, a] = p_e[t, s, a] / p_b[t, s, a]
            mu.append(mu_t)
        return mu

    def simulate_trajectory(self):
        """Simulate one trajectory under behavior policy pi_b."""
        trajectory = []
        s = self.s0
        for t in range(self.T + 1):
            a = np.random.choice(self.num_actions, p=self.pi_b[s])
            r = self.R[s, a]
            trajectory.append((s, a, r))
            if t < self.T:
                s = np.random.choice(self.num_states, p=self.P[s, a])
        return trajectory

    def compute_eif(self, trajectory, rho, Q, V, mu):
        """Compute the efficient influence function for one trajectory."""
        phi = -rho
        terms = 0.0
        for t in range(self.T + 1):
            s_t, a_t, r_t = trajectory[t]
            terms += mu[t][s_t, a_t] * (r_t - Q[t][s_t, a_t])
            if t == 0:
                terms += V[0][s_t]
            else:
                s_prev, a_prev, _ = trajectory[t-1]
                terms += mu[t-1][s_prev, a_prev] * V[t][s_t]
        return phi + terms, terms


def run_simulation(mdp, n_trajectories, seed=None):
    """Run multiple simulations and print summary of EIF estimates."""
    Q, V = mdp.compute_values()
    mu = mdp.compute_mu()
    rho_true = V[0][mdp.s0]
    eif_vals, term_vals = [], []
    if seed is not None:
        np.random.seed(seed)
    for _ in range(n_trajectories):
        traj = mdp.simulate_trajectory()
        phi, terms = mdp.compute_eif(traj, rho_true, Q, V, mu)
        eif_vals.append(phi)
        term_vals.append(terms)
    print(f"True rho^{{pi^e}}: {rho_true:.4f}")
    print(f"Average EIF: {np.mean(eif_vals):.4f}")
    print(f"Estimated rho^{{pi^e}}: {np.mean(term_vals):.4f}")
    # return data for further analysis
    return np.array(eif_vals), np.array(term_vals)
