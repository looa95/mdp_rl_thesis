import numpy as np
import sqlite3
from typing import List, Tuple, Dict, Optional

class HighwayMultiAgentEnvSQL:
    """
    Multi-agent highway Markov game with DB-backed SNR and lane/direction geometry.

    Geometry:
      - Ring road in x with length L; lanes are parallel lines at y = lane_index * lane_width.
      - You can set different movement senses per lane: +1 (forward) or -1 (reverse).
      - Agents have (x, lane) state; y is implied by the lane center.

    Dynamics:
      - Each step, every agent chooses resource a in {0,1,2}.
      - Success/collision decided by SNR from SQLite DB using (resource, tx_power_dbm, Euclidean distance).
      - Kinematics: x <- (x + dir * v * dt) mod L. v has small Gaussian jitter; lanes don't change in this minimal model
        (easy to extend with lane-change actions later).

    Labels: 0=idle, 1=success, 2=collision, 3=unavailable.
    """

    def __init__(
        self,
        num_agents: int = 20,
        T: int = 100,
        road_length: float = 1000.0,
        v_min: float = 18.0,
        v_max: float = 33.0,
        dt: float = 1.0,
        db_path: str = "radio_model.db",
        interference_max_range: float = 500.0,
        p_block_unavailable: float = 0.0,
        seed: Optional[int] = 20,
        tx_power_dbm: Optional[np.ndarray] = None,
        # --- geometry params ---
        num_lanes_total: int = 4,
        lane_width: float = 4,
        lane_directions: Optional[List[int]] = None,  # e.g., [+1,+1,-1,-1]; if None, first half +1, rest -1
        store_history: bool = True,
        #radio propagation params
        noise_mw: float = 1e-9,  # thermal noise in mW
        pathloss_n_los: float = 2.0,  # path-loss exponent for LOS
        pathloss_n_nlos: float = 3.5,  # path-loss exponent for NLOS
        desired_link_distance_m: float = 10.0, # desired signal link distance in meters
        # -- decision rules --
        snr_success_mode: str = "deterministic", #deterministic or stochastic
        snr_prob_alpha: float = .8, #only for stochastic mode, controls steepness of logistic success probability
        # -- init spacing
        min_initial_gap: float = 2.0  # meters ≥2 m within each lane at t=0



    ) -> None:
        self.N = num_agents
        self.T = T
        self.L = road_length
        self.v_min = v_min
        self.v_max = v_max
        self.dt = dt
        self.interf_range = interference_max_range
        self.p_block_unavailable = p_block_unavailable
        self.db_path = db_path
        self.rng = np.random.default_rng(seed)
        self.R = 3  # actions/resources
        self.noise_mw = noise_mw
        self.pathloss_n_los = pathloss_n_los
        self.pathloss_n_nlos = pathloss_n_nlos
        self.desired_link_distance_m = desired_link_distance_m
        self.snr_success_mode = snr_success_mode
        self.snr_prob_alpha = snr_prob_alpha
        self.min_initial_gap = min_initial_gap

        min_gap=self.min_initial_gap
        # --- Lanes / directions ---
        assert num_lanes_total >= 1
        self.num_lanes_total = num_lanes_total
        self.lane_width = lane_width
        if lane_directions is None:
            half = num_lanes_total // 2
            lane_directions = [+1]*max(1, half) + [-1]*(num_lanes_total - max(1, half))
        assert len(lane_directions) == num_lanes_total
        assert all(d in (-1, +1) for d in lane_directions)
        self.lane_directions = np.array(lane_directions, dtype=int)
        self.lane_centers_y = np.arange(num_lanes_total, dtype=float) * lane_width

        # --- Initial positions/speeds/lanes with min gap ---
        self.x0 = np.full(self.N, np.nan)
        self.v0 = self.rng.uniform(self.v_min, self.v_max, size=self.N)

        # random lanes with mild balance
        self.lane0 = self.rng.integers(low=0, high=self.num_lanes_total, size=self.N)

        min_gap = 2.0  # meters
        for lane in range(self.num_lanes_total):
            idx = np.where(self.lane0 == lane)[0]
            if idx.size == 0:
                continue
            # equally spaced within lane
            gaps = self.L / idx.size
            positions = np.array([k * gaps for k in range(idx.size)], dtype=float)
            # add small jitter but keep min_gap
            jitter = self.rng.uniform(-0.2*gaps, 0.2*gaps, size=idx.size)
            positions = (positions + jitter) % self.L
            positions.sort()
            # enforce min_gap
            for j in range(1, len(positions)):
                if positions[j] - positions[j-1] < min_gap:
                    positions[j] = positions[j-1] + min_gap
            # wrap last vs first
            if (self.L + positions[0] - positions[-1]) % self.L < min_gap:
                positions[-1] = (positions[0] - min_gap) % self.L
            self.x0[idx] = positions

        # Per-agent TX power (dBm)
        if tx_power_dbm is None:
            self.tx_power_dbm = self.rng.integers(low=18, high=34, size=self.N)  # 18..33
        else:
            self.tx_power_dbm = np.asarray(tx_power_dbm).astype(int)
            assert self.tx_power_dbm.shape == (self.N,)

        # Labels
        self.S_labels = ["idle", "success", "collision", "unavailable"]

        # Threshold from DB
        self.snr_success_thresh_db = self._get_snr_threshold()

        # History buffers
        self.store_history = store_history
        self._reset_history()

    # ---------- SQLite helpers ----------
    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _get_snr_threshold(self) -> float:
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute("SELECT value FROM params WHERE key='snr_success_thresh_db'")
                row = cur.fetchone()
                return float(row[0]) if row else 5.0
        except Exception:
            return 5.0

    def _snr_lookup(self, resource: int, tx_power_dbm: int, distance_m: float) -> Optional[float]:
        """
        Return SNR (dB) from snr_lookup by (resource, distance bin, power bin or nearest).
        """
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                # exact power bin
                cur.execute(
                    """
                    SELECT snr_db FROM snr_lookup
                    WHERE resource=? AND tx_power_dbm=? AND ? >= dist_min AND ? < dist_max
                    LIMIT 1
                    """,
                    (int(resource), int(tx_power_dbm), float(distance_m), float(distance_m))
                )
                row = cur.fetchone()
                if row:
                    return float(row[0])
                # nearest power bin
                cur.execute(
                    """
                    SELECT snr_db, ABS(tx_power_dbm - ?) AS delta
                    FROM snr_lookup
                    WHERE resource=? AND ? >= dist_min AND ? < dist_max
                    ORDER BY delta ASC
                    LIMIT 1
                    """,
                    (int(tx_power_dbm), int(resource), float(distance_m), float(distance_m))
                )
                row = cur.fetchone()
                if row:
                    return float(row[0])
        except Exception:
            pass
        return None
    
    def compute_tx_power_mw(self, tx_power_dbm:float) -> float:
        """
        Convert transmit power from dBm to mW.
        """
        return 10 ** ((tx_power_dbm ) / 10) 
    
    # ---------- 3GPP LOS/NLOSv probabilities ----------
    
    def p_los_highway(self, distance_m: float) -> float:
        """
        Probability of LOS (Line of Sight) according to 3GPP TR 38.901 Table 6.2-1 (Highway).
        """
        d = float(distance_m)
        if d <= 475.0:
            a, b, c = 2.1013e-6, -0.002, 1.0193
            p_los = a * d**2 + b * d + c
            return min(1.0, max(0.0, p_los))
        else:
            p_los = 0.54 - 0.001 * (d - 475.0)
            return max(0.0, min(1.0, p_los))

    def p_nlos_highway(self, distance_m: float) -> float:
        """
        Probability of NLOSv (non-LOS for vehicles).
        """
        return 1.0 - self.p_los_highway(distance_m)
    

    def compute_path_loss(self, distance_m: float, n_los: float = 2.0, n_nlos: float = 3.5) -> float:
        """
        Distance-based path loss with stochastic LOS/NLOS according to 3GPP 38.901.
        """
        p_los = self.p_los_highway(distance_m)
        if self.rng.random() < p_los:
            n = n_los
        else:
            n = n_nlos
        return 1.0 / (distance_m**n + 1e-9)
    
    def _path_loss_gain_sample(self, d_m: float, n_los: float, n_nlos: float) -> float:
        """
        Sample LOS/NLOS according to 3GPP highway model and return linear path-loss gain ~ 1/d^n.
        This is stochastic (uses self.rng).
        """
        # clamp very small distances to avoid blow-up
        d = max(1e-3, float(d_m))

        # sample LOS vs NLOS
        if self.rng.random() < self.p_los_highway(d):
            n = n_los
        else:
            n = n_nlos

        # simple power-law gain; you can replace by log-distance with reference if you like
        return 1.0 / (d ** n + 1e-12)
    
    def compute_sinr_linear(self, tx_idx: int, actions: np.ndarray) -> float:
        """
        SINR = S / (I + N) in linear units (mW/mW).
        - Desired received power S: uses a configurable desired-link distance.
        - Interference I: sum of *all* co-channel interferers within interference_max_range.
        - Thermal noise N: self.noise_mw (mW).
        - LOS/NLOS for each link is sampled *every call* from the 3GPP highway P(LOS).
        """
        ai = int(actions[tx_idx])
        # tx position (proxy for Rx position in this simplified model)
        y_now = self._lane_y(self.lane)
        xi, yi = float(self.x[tx_idx]), float(y_now[tx_idx])

        # --- Desired signal ---
        pt_dbm = float(self.tx_power_dbm[tx_idx])
        pt_mw  = self.compute_tx_power_mw(pt_dbm)
        # Use a small configurable desired-link distance (proxy to paired Rx)
        d_sig = float(self.desired_link_distance_m)
        g_sig = self._path_loss_gain_sample(d_sig, self.pathloss_n_los, self.pathloss_n_nlos)
        S = pt_mw * g_sig

        # --- Interference (sum over all co-channel interferers within range) ---
        I = 0.0
        for j in range(self.N):
            if j == tx_idx or int(actions[j]) != ai:
                continue
            xj, yj = float(self.x[j]), float(y_now[j])
            d_ij = self._ring_distance_2d(xi, yi, xj, yj)
            if d_ij > self.interf_range:
                continue
            ptj_dbm = float(self.tx_power_dbm[j])
            ptj_mw  = self.compute_tx_power_mw(ptj_dbm)
            g_ij = self._path_loss_gain_sample(d_ij, self.pathloss_n_los, self.pathloss_n_nlos)
            I += ptj_mw * g_ij

        # --- Noise ---
        N = float(self.noise_mw)

        return S / (I + N + 1e-18)

    def compute_sinr_db(self, tx_idx: int, actions: np.ndarray) -> float:
        sinr_lin = self.compute_sinr_linear(tx_idx, actions)
        return 10.0 * np.log10(max(1e-18, sinr_lin))



    # ---------- Helpers: geometry, distances, history ----------
    def _lane_y(self, lane_idx: np.ndarray) -> np.ndarray:
        return self.lane_centers_y[lane_idx]

    def _ring_dx(self, xi, xj) -> float:
        """
        Signed shortest dx on the ring (for completeness).
        """
        dx = (xj - xi + self.L/2) % self.L - self.L/2
        return float(dx)

    def _ring_distance_2d(self, xi, yi, xj, yj) -> float:
        """
        Euclidean distance using ring metric in x and direct in y.
        """
        dx = abs((xi - xj + self.L/2) % self.L - self.L/2)
        dy = abs(yi - yj)
        return float(np.hypot(dx, dy))

    def _reset_history(self):
        self.hist = {
            "x": [],            # shape (t, N)
            "lane": [],         # shape (t, N)
            "y": [],            # derived from lane
            "v": [],            # speeds
            "labels": [],       # 0..3
            "actions": []       # 0..2 (per step, so length t-1 compared to x)
        }

    def _log_timestep(self, x, lane, v, labels, actions=None):
        if not self.store_history:
            return
        y = self._lane_y(lane)
        self.hist["x"].append(x.copy())
        self.hist["lane"].append(lane.copy())
        self.hist["y"].append(y.copy())
        self.hist["v"].append(v.copy())
        self.hist["labels"].append(labels.copy())
        if actions is not None:
            self.hist["actions"].append(actions.copy())

    # ---------- Environment dynamics ----------
    def reset(self):
        self.x = self.x0.copy()
        self.v = self.v0.copy()
        self.lane = self.lane0.copy()
        self.s = np.zeros(self.N, dtype=int)  # idle
        self._reset_history()
        # log t=0
        self._log_timestep(self.x, self.lane, self.v, self.s, actions=None)
        return self._obs()

    def _obs(self) -> Dict[str, np.ndarray]:
        return {
            "x": self.x.copy(),
            "y": self._lane_y(self.lane).copy(),
            "lanes": self.lane.copy(),
            "velocities": self.v.copy(),
            "labels": self.s.copy(),
            "tx_power_dbm": self.tx_power_dbm.copy(),
            "lane_directions": self.lane_directions.copy(),
            "lane_width": np.array([self.lane_width], dtype=float)
        }

    def step(self, actions: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray, bool, Dict]:
        assert actions.shape[0] == self.N
        assert np.all((0 <= actions) & (actions < self.R))

        unavailable = (np.random.random(self.N) < self.p_block_unavailable)

        rewards = np.zeros(self.N, dtype=float)
        labels = np.zeros(self.N, dtype=int)

        # --- Interference / SNR success-vs-collision ---
        # Use Euclidean distance in 2D (ring in x, fixed y by lane).
        y_now = self._lane_y(self.lane)

        for i in range(self.N):
            if unavailable[i]:
                labels[i] = 3  # unavailable
                rewards[i] = 0.0
                continue

            ai = int(actions[i])

            # interferers on same resource within range
            interferers = []
            for j in range(self.N):
                if j == i or actions[j] != ai:
                    continue
                d_ij = self._ring_distance_2d(self.x[i], y_now[i], self.x[j], y_now[j])
                if d_ij <= self.interf_range:
                    interferers.append((j, d_ij))

            # --- Compute SNR/SINR --
            sinr_db = self.compute_sinr_db(i, actions)

            if self.snr_success_mode == 'deterministic':
                if sinr_db >= self.snr_success_thresh_db:
                    labels[i] = 1; rewards[i] = 10.0
                else:
                    labels[i] = 2; rewards[i] = -5.0
            else:
                # Stochastic: Bernoulli with P_success = sigmoid(alpha*(sinr_db - thresh))
                alpha = float(self.snr_prob_alpha)
                p_success = 1.0 / (1.0 + np.exp(-alpha * (sinr_db - self.snr_success_thresh_db)))
                if self.rng.random() < p_success:
                    labels[i] = 1; rewards[i] = 10.0
                else:
                    labels[i] = 2; rewards[i] = -5.0
                    # --- Kinematics: x progression by lane direction ---
                    lane_dir = self.lane_directions[self.lane]  # +1 or -1 per agent
                    self.x = (self.x + lane_dir * self.v * self.dt) % self.L
                    self.v = np.clip(self.v + self.rng.normal(0, 0.2, size=self.N), self.v_min, self.v_max)
                    self.s = labels

        obs = self._obs()
        done = False
        info = {}
        self._log_timestep(self.x, self.lane, self.v, self.s, actions=actions)
        return obs, rewards, done, info

    def rollout(self, policies: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Roll out for T steps.
        policies: list of per-agent (4,3) arrays: pi(a|label). If length==1, share across agents.
        Returns (labels_hist, actions_hist, rewards_hist).
        """
        if len(policies) == 1:
            policies = [policies[0] for _ in range(self.N)]
        assert len(policies) == self.N
        for p in policies:
            assert p.shape == (4, 3) and np.allclose(p.sum(axis=1), 1.0)

        self.reset()
        labels_hist = [self.s.copy()]
        actions_hist = []
        rewards_hist = []

        for _ in range(self.T+1):
            acts = np.zeros(self.N, dtype=int)
            for i in range(self.N):
                pi = policies[i][ self.s[i] ]
                acts[i] = np.random.choice(3, p=pi)
            obs, r, done, info = self.step(acts)
            actions_hist.append(acts.copy())
            rewards_hist.append(r.copy())
            labels_hist.append(obs["labels"].copy())

        return labels_hist, actions_hist, rewards_hist

    # ---------- NEW: trajectory & geometry utilities ----------
    def get_history(self) -> Dict[str, np.ndarray]:
        """
        Returns a dict of arrays with shapes:
          - x: (T+1, N)
          - y: (T+1, N)
          - lane: (T+1, N)
          - v: (T+1, N)
          - labels: (T+1, N)
          - actions: (T+1, N) except the first row is not defined at t=0; we store length T+1 with first row NaNs here.
        """
        if not self.store_history or len(self.hist["x"]) == 0:
            return {}
        # pad actions to align with x timeline (T+1)
        actions = self.hist["actions"]
        if len(actions) < len(self.hist["x"]):
            pad = np.full((1, self.N), np.nan)
            A = np.concatenate([pad] + [np.array(a, ndmin=2) for a in actions], axis=0)
        else:
            A = np.array(actions)
        return {
            "x": np.array(self.hist["x"]),
            "y": np.array(self.hist["y"]),
            "lane": np.array(self.hist["lane"]),
            "v": np.array(self.hist["v"]),
            "labels": np.array(self.hist["labels"]),
            "actions": A
        }

    def summarize_layout(self) -> Dict[str, np.ndarray]:
        """
        Basic geometry summary: lane centers, directions, lane counts at t=0.
        """
        counts = np.bincount(self.lane0, minlength=self.num_lanes_total)
        return {
            "num_lanes_total": np.array([self.num_lanes_total]),
            "lane_width": np.array([self.lane_width]),
            "lane_centers_y": self.lane_centers_y.copy(),
            "lane_directions": self.lane_directions.copy(),
            "initial_lane_counts": counts
        }

    def spacing_stats(self) -> Dict[str, np.ndarray]:
        """
        Per-lane nearest-neighbor longitudinal gaps (in x) over time.
        Returns:
          - mean_gap_per_lane: (T+1, num_lanes)
          - min_gap_per_lane:  (T+1, num_lanes)
        """
        if not self.store_history or len(self.hist["x"]) == 0:
            return {}
        X = np.array(self.hist["x"])  # (T+1, N)
        LANE = np.array(self.hist["lane"])  # (T+1, N)
        T1, N = X.shape
        L = self.L
        nl = self.num_lanes_total
        mean_gap = np.full((T1, nl), np.nan)
        min_gap = np.full((T1, nl), np.nan)
        for t in range(T1):
            for ell in range(nl):
                idx = np.where(LANE[t] == ell)[0]
                if idx.size < 2:
                    continue
                xs = np.sort(X[t, idx])
                # ring gaps within this lane
                diffs = np.diff(xs, append=xs[0] + L)
                mean_gap[t, ell] = diffs.mean()
                min_gap[t, ell] = diffs.min()
        return {"mean_gap_per_lane": mean_gap, "min_gap_per_lane": min_gap}


# ---------------------------
# Thesis policies
# ---------------------------

def behavior_policy() -> np.ndarray:
    """π_b(a|s): uniform over actions for every label."""
    return np.ones((4,3)) / 3.0

def evaluation_policy() -> np.ndarray:
    """π_e(a|s): biased towards action 0 on {idle, success}; exploratory on {collision, unavailable}."""
    p = np.zeros((4,3))
    p[0] = np.array([0.7, 0.2, 0.1])  # idle
    p[1] = np.array([0.7, 0.2, 0.1])  # success
    p[2] = np.array([0.5, 0.3, 0.2])  # collision
    p[3] = np.array([0.5, 0.3, 0.2])  # unavailable
    return p


# ---------------------------
# Convenience runner
# ---------------------------

def simulate_highway_multiagent_sql(
    N: int = 20, T: int = 100, seed: int = 20,
    db_path: str = "radio_model.db",
    interference_max_range: float = 500.0,
    num_lanes_total: int = 4,
    lane_width: float = 3.75,
    lane_directions: Optional[List[int]] = None,
    store_history: bool = True
) -> dict:
    """
    Runs one rollout with behavior policy and one with evaluation policy.
    Returns time series of avg rewards and label rates, raw histories, and geometry summaries.
    """
    env = HighwayMultiAgentEnvSQL(
        num_agents=N, T=T, seed=seed, db_path=db_path,
        interference_max_range=interference_max_range,
        num_lanes_total=num_lanes_total, lane_width=lane_width,
        lane_directions=lane_directions, store_history=store_history
    )
    pi_b = behavior_policy()
    pi_e = evaluation_policy()

    labels_b, actions_b, rewards_b = env.rollout([pi_b])
    hist_b = env.get_history()
    layout = env.summarize_layout()
    spacing = env.spacing_stats()

    labels_e, actions_e, rewards_e = env.rollout([pi_e])
    hist_e = env.get_history()

    avg_rew_b = np.array([r.mean() for r in rewards_b])
    avg_rew_e = np.array([r.mean() for r in rewards_e])

    def rates(labels_hist):
        labs = np.array(labels_hist[1:])
        succ = (labs == 1).mean(axis=1)
        coll = (labs == 2).mean(axis=1)
        unav = (labs == 3).mean(axis=1)
        return succ, coll, unav

    succ_b, coll_b, unav_b = rates(labels_b)
    succ_e, coll_e, unav_e = rates(labels_e)

    return {
        "avg_rew_b": avg_rew_b, "avg_rew_e": avg_rew_e,
        "succ_b": succ_b, "succ_e": succ_e,
        "coll_b": coll_b, "coll_e": coll_e,
        "unav_b": unav_b, "unav_e": unav_e,
        "labels_b": labels_b, "actions_b": actions_b, "rewards_b": rewards_b,
        "labels_e": labels_e, "actions_e": actions_e, "rewards_e": rewards_e,
        "history_b": hist_b, "history_e": hist_e,
        "layout": layout, "spacing": spacing
    }
