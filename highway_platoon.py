import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional

# ============================================================
# Clean Highway Multi-Agent (CSV BLER only, 2 actions, platoon)
# ============================================================

class HighwayMultiAgentClean:
    """
    Minimal highway multi-agent environment with:
      - Ring road geometry; fixed lanes with configurable directions.
      - Two actions: choose resource 0 or 1 (R=2). Transmission outcome is Success/Collision.
      - Three labels (states) used for policy conditioning:
          0 = idle (only at t=0)
          1 = success
          2 = collision
      - CSV-only radio model: reads BLER (block error rate) vs SINR (dB), per resource.
        Interpolates BLER; p_success = 1 - BLER. No SQLite. No 'unavailable' state.

    Transition logic per agent i (given joint actions a):
        SINR_i -> BLER_i -> p_succ_i = 1 - BLER_i
        s_{t+1} = 1 (success) w.p. p_succ_i
                 = 2 (collision) otherwise
        (idle occurs only at reset t=0)
    """

    # ------------------ construction ------------------
    def __init__(
        self,
        num_agents: int = 20,
        T: int = 100,
        road_length: float = 1000.0,
        v_min: float = 18.0,
        v_max: float = 33.0,
        dt: float = 1.0,
        *,
        bler_csv_path: str,
        interference_max_range: float = 500.0,
        seed: Optional[int] = 20,
        # geometry
        num_lanes_total: int = 4,
        lane_width: float = 3.75,
        lane_directions: Optional[List[int]] = None,  # +1 or -1 per lane
        # radio / propagation
        noise_mw: float = 1e-9,
        pathloss_n_los: float = 2.0,
        pathloss_n_nlos: float = 3.5,
        desired_link_distance_m: float = 10.0,
        # init spacing
        min_initial_gap: float = 2.0,
        # tx power per agent (dBm), if None draw 18..33 dBm
        tx_power_dbm: Optional[np.ndarray] = None,
        store_history: bool = True,
    ) -> None:
        self.N = int(num_agents)
        self.T = int(T)
        self.L = float(road_length)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.dt = float(dt)

        self.interf_range = float(interference_max_range)
        self.noise_mw = float(noise_mw)
        self.pathloss_n_los = float(pathloss_n_los)
        self.pathloss_n_nlos = float(pathloss_n_nlos)
        self.desired_link_distance_m = float(desired_link_distance_m)

        self.num_lanes_total = int(num_lanes_total)
        self.lane_width = float(lane_width)

        if lane_directions is None:
            half = max(1, self.num_lanes_total // 2)
            lane_directions = [+1] * half + [-1] * (self.num_lanes_total - half)
        assert len(lane_directions) == self.num_lanes_total
        assert all(d in (-1, +1) for d in lane_directions)
        self.lane_directions = np.array(lane_directions, dtype=int)
        self.lane_centers_y = np.arange(self.num_lanes_total, dtype=float) * self.lane_width

        self.rng = np.random.default_rng(seed)

        # actions/resources: 0 or 1
        self.R = 2

        # labels: 0=idle (t=0 only), 1=success, 2=collision
        self.S_labels = ["idle", "success", "collision"]

        # Initial positions/velocities/lanes with minimal spacing per lane
        self.x0 = np.full(self.N, np.nan, dtype=float)
        self.v0 = self.rng.uniform(self.v_min, self.v_max, size=self.N)
        self.lane0 = self.rng.integers(low=0, high=self.num_lanes_total, size=self.N)
        self._init_positions_with_min_gap(min_initial_gap)

        # Per-agent TX power (dBm)
        if tx_power_dbm is None:
            self.tx_power_dbm = self.rng.integers(low=18, high=34, size=self.N)  # 18..33 dBm
        else:
            self.tx_power_dbm = np.asarray(tx_power_dbm).astype(int)
            assert self.tx_power_dbm.shape == (self.N,)

        # CSV BLER curves (required)
        self._load_bler_from_csv(bler_csv_path)

        # runtime state & history
        self.store_history = bool(store_history)
        self._reset_history()
        self.reset()

    # ------------------ CSV BLER handling ------------------
    def _load_bler_from_csv(self, path: str) -> None:
        """
        Expect wide or long CSV. Two accepted formats:

        (A) Long form (preferred):
            resource,snr_db,bler
        (B) Simple single-resource curve:
            sinr_db,mcs_0_bler
        In (B) we assume resource=0; to use 2 resources, duplicate file or add a column externally.
        """
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]

        if {"resource", "snr_db", "bler"}.issubset(df.columns):
            P = df[["resource", "snr_db", "bler"]].copy()
        elif {"sinr_db", "mcs_0_bler"}.issubset(df.columns):
            P = df.rename(columns={"sinr_db": "snr_db", "mcs_0_bler": "bler"}).copy()
            P["resource"] = 0
        else:
            raise ValueError(
                "CSV must be either (resource,snr_db,bler) or (sinr_db,mcs_0_bler)."
            )

        # sanitize & sort
        P["resource"] = P["resource"].astype(int)
        P["snr_db"] = P["snr_db"].astype(float)
        P["bler"] = np.clip(P["bler"].astype(float), 0.0, 1.0)
        P = P.sort_values(["resource", "snr_db"]).reset_index(drop=True)

        # split by resource for fast lookup
        self._bler_points = {
            r: grp[["snr_db", "bler"]].to_numpy()
            for r, grp in P.groupby("resource", sort=True)
        }
        # Ensure we have both resources (0,1). If only one provided, reuse it for the missing one.
        if 0 not in self._bler_points and 1 in self._bler_points:
            self._bler_points[0] = self._bler_points[1]
        if 1 not in self._bler_points and 0 in self._bler_points:
            self._bler_points[1] = self._bler_points[0]

    def _bler_interp(self, resource: int, snr_db: float) -> float:
        """Linear interpolate BLER at snr_db for a given resource (0/1)."""
        tbl = self._bler_points.get(int(resource))
        if tbl is None or tbl.shape[0] == 0:
            return 1.0  # pessimistic if table missing
        xs = tbl[:, 0]
        ys = tbl[:, 1]
        # left/right neighbors
        i = np.searchsorted(xs, snr_db, side="left")
        if i == 0:
            return float(ys[0])
        if i >= len(xs):
            return float(ys[-1])
        x0, y0 = xs[i - 1], ys[i - 1]
        x1, y1 = xs[i], ys[i]
        if x1 == x0:
            return float(0.5 * (y0 + y1))
        t = (snr_db - x0) / (x1 - x0)
        return float(np.clip(y0 + t * (y1 - y0), 0.0, 1.0))

    # ------------------ geometry ------------------
    def _init_positions_with_min_gap(self, min_gap: float) -> None:
        for lane in range(self.num_lanes_total):
            idx = np.where(self.lane0 == lane)[0]
            if idx.size == 0:
                continue
            gaps = self.L / idx.size
            positions = np.array([k * gaps for k in range(idx.size)], dtype=float)
            jitter = self.rng.uniform(-0.2 * gaps, 0.2 * gaps, size=idx.size)
            positions = (positions + jitter) % self.L
            positions.sort()
            for j in range(1, len(positions)):
                if positions[j] - positions[j - 1] < min_gap:
                    positions[j] = positions[j - 1] + min_gap
            if (self.L + positions[0] - positions[-1]) % self.L < min_gap:
                positions[-1] = (positions[0] - min_gap) % self.L
            self.x0[idx] = positions

    def _lane_y(self, lane_idx: np.ndarray) -> np.ndarray:
        return self.lane_centers_y[lane_idx]

    def _ring_distance_2d(self, xi, yi, xj, yj) -> float:
        dx = abs((xi - xj + self.L / 2) % self.L - self.L / 2)
        dy = abs(yi - yj)
        return float(np.hypot(dx, dy))

    # ------------------ radio ------------------
    @staticmethod
    def _dbm_to_mw(p_dbm: float) -> float:
        return 10.0 ** (p_dbm / 10.0)

    def _p_los_highway(self, d_m: float) -> float:
        d = float(max(0.0, d_m))
        if d <= 475.0:
            a, b, c = 2.1013e-6, -0.002, 1.0193
            p = a * d ** 2 + b * d + c
            return float(np.clip(p, 0.0, 1.0))
        return float(np.clip(0.54 - 0.001 * (d - 475.0), 0.0, 1.0))

    def _sample_pathloss_gain(self, d_m: float) -> float:
        # LOS vs NLOS exponent sample
        n = self.pathloss_n_los if (self.rng.random() < self._p_los_highway(d_m)) else self.pathloss_n_nlos
        d = max(1e-3, float(d_m))
        return 1.0 / (d ** n + 1e-12)

    def _compute_sinr_db(self, i: int, actions: np.ndarray) -> float:
        """Compute SINR (dB) for agent i given joint actions over two resources."""
        ai = int(actions[i])
        xi, yi = float(self.x[i]), float(self._lane_y(self.lane)[i])

        # desired link
        pt_mw = self._dbm_to_mw(float(self.tx_power_dbm[i]))
        g_sig = self._sample_pathloss_gain(float(self.desired_link_distance_m))
        S = pt_mw * g_sig

        # interference: sum of j sharing resource ai
        I = 0.0
        for j in range(self.N):
            if j == i or int(actions[j]) != ai:
                continue
            xj, yj = float(self.x[j]), float(self._lane_y(self.lane)[j])
            d_ij = self._ring_distance_2d(xi, yi, xj, yj)
            if d_ij > self.interf_range:
                continue
            ptj_mw = self._dbm_to_mw(float(self.tx_power_dbm[j]))
            g_ij = self._sample_pathloss_gain(d_ij)
            I += ptj_mw * g_ij

        N = float(self.noise_mw)
        sinr_lin = S / (I + N + 1e-18)
        return 10.0 * np.log10(max(1e-18, sinr_lin))

    # ------------------ history ------------------
    def _reset_history(self) -> None:
        self.hist = {"x": [], "lane": [], "y": [], "v": [], "labels": [], "actions": []}

    def _log(self, actions: Optional[np.ndarray]) -> None:
        if not self.store_history:
            return
        self.hist["x"].append(self.x.copy())
        self.hist["lane"].append(self.lane.copy())
        self.hist["y"].append(self._lane_y(self.lane).copy())
        self.hist["v"].append(self.v.copy())
        self.hist["labels"].append(self.s.copy())
        if actions is not None:
            self.hist["actions"].append(actions.copy())

    def get_history(self) -> Dict[str, np.ndarray]:
        if not self.store_history or len(self.hist["x"]) == 0:
            return {}
        A = np.array(self.hist["actions"]) if self.hist["actions"] else np.full((0, self.N), np.nan)
        if A.shape[0] < len(self.hist["x"]):
            pad = np.full((1, self.N), np.nan)
            A = np.concatenate([pad, A], axis=0)
        return {
            "x": np.array(self.hist["x"]),
            "y": np.array(self.hist["y"]),
            "lane": np.array(self.hist["lane"]),
            "v": np.array(self.hist["v"]),
            "labels": np.array(self.hist["labels"]),
            "actions": A,
        }

    # ------------------ env API ------------------
    def reset(self) -> Dict[str, np.ndarray]:
        self.x = self.x0.copy()
        self.v = self.v0.copy()
        self.lane = self.lane0.copy()
        self.s = np.zeros(self.N, dtype=int)  # idle at t=0
        self._reset_history()
        self._log(actions=None)
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
            "lane_width": np.array([self.lane_width], dtype=float),
        }

    def step(self, actions: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray, bool, Dict]:
        assert actions.shape == (self.N,)
        assert np.all((0 <= actions) & (actions < self.R))

        labels = np.zeros(self.N, dtype=int)
        rewards = np.zeros(self.N, dtype=float)

        # transmission outcomes (success/collision)
        for i in range(self.N):
            sinr_db = self._compute_sinr_db(i, actions)
            ai = int(actions[i])
            bler = self._bler_interp(ai, sinr_db)
            p_success = 1.0 - bler
            ok = (self.rng.random() < p_success)
            labels[i] = 1 if ok else 2
            rewards[i] = 1.0 if ok else -1.0

        # kinematics
        lane_dir = self.lane_directions[self.lane]
        self.x = (self.x + lane_dir * self.v * self.dt) % self.L
        self.v = np.clip(self.v + self.rng.normal(0, 0.2, size=self.N), self.v_min, self.v_max)
        self.s = labels

        obs = self._obs()
        done = False
        info = {}
        self._log(actions)
        return obs, rewards, done, info

    def rollout(self, policies: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        policies: list of per-agent (3,2) arrays: pi(a|label) for labels {0,1,2} and actions {0,1}.
        If len(policies)==1, share across agents.
        Returns (labels_hist, actions_hist, rewards_hist).
        """
        if len(policies) == 1:
            policies = [policies[0] for _ in range(self.N)]
        assert len(policies) == self.N
        for p in policies:
            assert p.shape == (3, 2) and np.allclose(p.sum(axis=1), 1.0)

        self.reset()
        labels_hist = [self.s.copy()]
        actions_hist, rewards_hist = [], []

        for _ in range(self.T + 1):
            acts = np.zeros(self.N, dtype=int)
            for i in range(self.N):
                row = policies[i][self.s[i]]
                acts[i] = self.rng.choice(2, p=row)
            _, r, _, _ = self.step(acts)
            actions_hist.append(acts.copy())
            rewards_hist.append(r.copy())
            labels_hist.append(self.s.copy())

        return labels_hist, actions_hist, rewards_hist

    # ------------------ layout & spacing (optional) ------------------
    def summarize_layout(self) -> Dict[str, np.ndarray]:
        counts = np.bincount(self.lane0, minlength=self.num_lanes_total)
        return {
            "num_lanes_total": np.array([self.num_lanes_total]),
            "lane_width": np.array([self.lane_width]),
            "lane_centers_y": self.lane_centers_y.copy(),
            "lane_directions": self.lane_directions.copy(),
            "initial_lane_counts": counts,
        }

    def spacing_stats(self) -> Dict[str, np.ndarray]:
        if not self.store_history or len(self.hist["x"]) == 0:
            return {}
        X = np.array(self.hist["x"])
        LANE = np.array(self.hist["lane"])
        T1, _ = X.shape
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
                diffs = np.diff(xs, append=xs[0] + L)
                mean_gap[t, ell] = diffs.mean()
                min_gap[t, ell] = diffs.min()
        return {"mean_gap_per_lane": mean_gap, "min_gap_per_lane": min_gap}


# =========================
# Policy / runner utilities
# =========================

def deterministic_policy_2actions(action_by_label: Dict[int, int]) -> np.ndarray:
    """
    Build a (3,2) policy deterministic per label (0..2) over actions {0,1}.
    Unspecified labels default to action 0.
    """
    P = np.zeros((3, 2))
    for s in range(3):
        a = int(action_by_label.get(s, 0))
        assert a in (0, 1)
        P[s, a] = 1.0
    return P

def mixed_policy_2actions(rows: Dict[int, Tuple[float, float]]) -> np.ndarray:
    """
    Build a (3,2) policy with custom probabilities for specified labels.
    Unspecified labels default to uniform [0.5, 0.5].
    """
    P = np.ones((3, 2)) * 0.5
    for s, probs in rows.items():
        v = np.asarray(probs, dtype=float)
        assert v.shape == (2,) and np.all(v >= 0) and v.sum() > 0
        P[s] = v / v.sum()
    return P

def make_platoon_policies(N: int, platoon_idx: List[int], pi_platoon: np.ndarray, pi_others: np.ndarray) -> List[np.ndarray]:
    """
    Return per-agent (3,2) policies: agents in platoon_idx use pi_platoon, others use pi_others.
    """
    out = [pi_others.copy() for _ in range(N)]
    if platoon_idx:
        idx = np.asarray(platoon_idx, dtype=int)
        assert np.all((0 <= idx) & (idx < N))
        for i in idx:
            out[i] = pi_platoon.copy()
    return out

def estimate_transition_matrix(labels_hist: List[np.ndarray], num_states: int = 3) -> np.ndarray:
    """
    Empirical T[s, s'] = P(s_{t+1}=s' | s_t=s) from history returned by env.rollout.
    """
    T = np.zeros((num_states, num_states), dtype=float)
    for t in range(len(labels_hist) - 1):
        s_now = labels_hist[t]
        s_next = labels_hist[t + 1]
        for s in range(num_states):
            idx = (s_now == s)
            if not np.any(idx):
                continue
            counts = np.bincount(s_next[idx], minlength=num_states)
            T[s] += counts
    # row normalize
    row_sums = T.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        T = np.divide(T, row_sums, out=np.zeros_like(T), where=row_sums > 0)
    return T

# --------------
# Example runner
# --------------
def simulate_with_platoon(
    *,
    N: int = 20,
    T: int = 100,
    seed: int = 20,
    bler_csv_path: str,
    platoon_indices: Optional[List[int]] = None,
    pi_platoon: Optional[np.ndarray] = None,
    pi_others: Optional[np.ndarray] = None,
    **env_kwargs,
) -> dict:
    """
    Run a single rollout using platoon vs others policies (no 'evaluation' policy).
    Returns histories, average rewards, and empirical transition matrix.
    """
    env = HighwayMultiAgentClean(
        num_agents=N, T=T, seed=seed, bler_csv_path=bler_csv_path, **env_kwargs
    )

    # defaults: platoon prefers resource 0 when last outcome was success; explore on collision
    if pi_platoon is None:
        pi_platoon = mixed_policy_2actions({
            0: (0.8, 0.2),   # idle -> favor res0
            1: (0.9, 0.1),   # after success -> stick to res0
            2: (0.4, 0.6),   # after collision -> shift to res1
        })
    if pi_others is None:
        pi_others = mixed_policy_2actions({
            0: (0.5, 0.5),   # idle -> neutral
            1: (0.6, 0.4),   # mild bias to res0
            2: (0.5, 0.5),   # neutral after collision
        })

    policies = (
        [pi_others] if not platoon_indices
        else make_platoon_policies(N, platoon_indices, pi_platoon, pi_others)
    )

    labels, actions, rewards = env.rollout(policies)
    hist = env.get_history()

    avg_rew = np.array([r.mean() for r in rewards])
    T_hat = estimate_transition_matrix(labels, num_states=3)

    return {
        "avg_rew": avg_rew,
        "labels_hist": labels,
        "actions_hist": actions,
        "rewards_hist": rewards,
        "history": hist,
        "transition_matrix": T_hat,
        "layout": env.summarize_layout(),
        "spacing": env.spacing_stats(),
    }
