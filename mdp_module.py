import numpy as np

class MDP:
    def __init__(self, num_states, num_actions, T, s0, P, R, pi_b, pi_e, gamma=1.0):
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
        - gamma: Discount factor (default is 1.0 for undiscounted MDP)
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.T = T
        self.s0 = s0
        self.P = P  # P[s, a, s'] = P(s' | s, a)
        self.R = R  # R[s, a] = reward
        self.pi_b = pi_b  # pi_b[s, a] = P(a | s)
        self.pi_e = pi_e  # pi_e[s, a] = P(a | s)
        self.gamma = gamma

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
                    Q[t][s, a] = self.R[s, a] + self.gamma *np.sum(self.P[s, a] * V[t + 1])
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
            mask = p_b[t] > 0
            mu_t[mask] = p_e[t][mask] / p_b[t][mask]
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
        cum_gamma = 1.0
        # Compute the terms in the EIF
        for t in range(self.T + 1):
            s_t, a_t, r_t = trajectory[t]
            terms += cum_gamma* mu[t][s_t, a_t] * (r_t - Q[t][s_t, a_t])
            if t == 0:
                terms += cum_gamma * V[0][s_t]
            else:
                s_prev, a_prev, _ = trajectory[t-1]
                terms += cum_gamma * mu[t-1][s_prev, a_prev] * V[t][s_t]
        return phi + terms, terms



def estimate_nuisances(self, trajectories):
    """Estimate Q_t and V_t using the given trajectories using least squares"""
    Q_hat = [np.zeros((self.num_states, self.num_actions)) for _ in range(self.T + 1)]
    V_hat = [np.zeros(self.num_states) for _ in range(self.T + 1)]
    counts = [np.zeros((self.num_states, self.num_actions)) for _ in range(self.T + 1)]
    
    for traj in trajectories:
        for t in range(self.T + 1):
            s, a, r = traj[t]
            counts[t][s, a] += 1
            future_reward = 0.0
            disc = 1.0
            for tt in range(t, self.T + 1):
                future_reward += disc * traj[tt][2]
                disc *= self.gamma
            Q_hat[t][s, a] += future_reward
            
    for t in range(self.T + 1):
        for s in range(self.num_states):
            for a in range(self.num_actions):
                if counts[t][s, a] > 0:
                    Q_hat[t][s, a] /= counts[t][s, a]
            V_hat[t][s] = np.sum(self.pi_e[s] * Q_hat[t][s])
    return Q_hat, V_hat

def estimate_mu(self, trajectories):
    """Estimate marginalized density ratios mu_t(s, a) = p_pi^e(t, s, a) / p_pi^b(t, s, a) """
    d_b_hat = np.zeros((self.T + 1, self.num_states, self.num_actions))
    n = len(trajectories)
    for traj in trajectories:
        for t in range(self.T + 1):
            s, a, _ = traj[t]
            d_b_hat[t, s, a] += 1 / n
    
    d_e_hat = np.zeros_like(d_b_hat)
    for traj in trajectories:
        cum_w = 1.0
        for t in range(self.T + 1):
            s, a, _ = traj[t]
            w = self.pi_e[s, a] / self.pi_b[s, a] if self.pi_b[s, a] > 0 else 0
            cum_w *= w
            d_e_hat[t, s, a] += cum_w / n
    
    mu_hat = []
    for t in range(self.T + 1):
        mu_t = np.zeros((self.num_states, self.num_actions))
        mask = d_b_hat[t] > 0
        mu_t[mask] = d_e_hat[t][mask] / d_b_hat[t][mask]
        mu_hat.append(mu_t)
    return mu_hat

def drl_estimator(self, trajectories):
    """ Double RL estimator with cross-fitting K=2 folds"""
    n = len(trajectories)
    if n < 2:
        return 0.0
    fold1 = trajectories[:n//2]
    fold2 = trajectories[n//2:]
    
    Q1, V1 = self.estimate_nuisances(fold1)
    mu1 = self.estimate_mu(fold1)
    Q2, V2 = self.estimate_nuisances(fold2)
    mu2 = self.estimate_mu(fold2)
    
    est = 0.0
    for traj in fold1:
        _, terms = self.compute_eif(traj, 0, Q2, V2, mu2)  # rho=0 to get terms as est component
        est += terms / n
    for traj in fold2:
        _, terms = self.compute_eif(traj, 0, Q1, V1, mu1)
        est += terms / n
    return est


def run_simulation(mdp, n_trajectories, seed=None):
    """Run multiple simulations and print summary of EIF estimates."""
    if seed is not None:
        np.random.seed(seed)
    # Compute necessary values
    Q_exact, V_exact = mdp.compute_values()
    mu_exact= mdp.compute_mu()
    rho_true = V_exact[0][mdp.s0]
    trajectories = [mdp.simulate_trajectory() for _ in range(n_trajectories)]
    eif_vals = []
    term_vals = []
    for traj in trajectories:
        phi, terms = mdp.compute_eif(traj, rho_true, Q_exact, V_exact, mu_exact)
        eif_vals.append(phi)
        term_vals.append(terms)
    
    eif_mean = np.mean(eif_vals)
    eif_var = np.var(eif_vals)
    rho_est_exact= np.mean(term_vals)
    rho_drl = mdp.drl_estimator(trajectories)

    # Variance over runs
    drl_estimates = [mdp.drl_estimator([mdp.simulate_trajectory() for _ in range(n_trajectories)]) for _ in range(10)]
    drl_var = np.var(drl_estimates)
    bound = eif_var / n_trajectories
    # DR demo
    Q_random = [np.random.uniform(-10, 10, (mdp.num_states, mdp.num_actions)) for _ in range(mdp.T + 1)]
    V_random = [np.sum(mdp.pi_e * Q_random[t], axis=1) for t in range(mdp.T + 1)]
    rho_miss_model = np.mean([mdp.compute_eif(traj, 0, Q_random, V_random, mu_exact)[1] for traj in trajectories])
    mu_random = [np.random.uniform(0, 2, (mdp.num_states, mdp.num_actions)) for _ in range(mdp.T + 1)]
    rho_miss_weights = np.mean([mdp.compute_eif(traj, 0, Q_exact, V_exact, mu_random)[1] for traj in trajectories]

    
    print(f"True rho^{{pi^e}}: {rho_true:.4f}")
    print(f"Average EIF: {eif_mean:.4f}")
    print(f"EIF variance: {eif_var:.4f}")
    print(f"Estimated rho from exact terms: {rho_est_exact:.4f}")
    print(f"DRL estimate (est nuisances): {rho_drl:.4f}")
    print(f"DRL variance (10 runs): {drl_var:.4f}")
    print(f"Efficiency bound (Var(EIF)/n): {bound:.4f}")
    print(f"DR with misspecified model (Q/V random, mu exact): {rho_miss_model:.4f}")
    print(f"DR with misspecified weights (mu random, Q/V exact): {rho_miss_weights:.4f}")
    
    return np.array(eif_vals), np.array(term_vals)
