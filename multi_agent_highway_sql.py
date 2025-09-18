import numpy as np
import sqlite3
from typing import List, Tuple, Dict, Optional
import pandas as pd

class HighwayMultiAgentEnvSQL:
    """
    Multi-agent highway Markov game with analytic SINR and 3GPP LOS/NLOS sampling.

    Geometry:
      - Ring road in x with length L; lanes are parallel lines at y = lane_index * lane_width.
      - You can set different movement senses per lane: +1 (forward) or -1 (reverse).
      - Agents have (x, lane) state; y is implied by the lane center.

    Dynamics:
      - Each step, every agent chooses resource a in {0,1,2}.
      - Radio: For each agent, compute SINR = S / (I + N) in linear units.
        * S uses a desired-link distance with stochastic LOS/NLOS (3GPP TR 38.901 highway P(LOS)).
        * I sums all co-channel interferers within `interference_max_range`, each with its own sampled LOS/NLOS.
        * N is thermal noise (mW).
      - Success / collision is decided by a probability fetched from SQLite:
        1) Try table `snr_success_prob` (binned by SNR dB, per resource).
        2) Else, linearly interpolate from `snr_success_curve` (points), per resource.
        If no DB mapping applies, default p_success = 0.0 (collision).
      - Kinematics: x <- (x + dir * v * dt) mod L. v has small Gaussian jitter; lanes are static in this minimal model.

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
        bler_table_path: Optional[str] = None,
        interference_max_range: float = 500.0,
        p_block_unavailable: float = 0.0,
        seed: Optional[int] = 20,
        tx_power_dbm: Optional[np.ndarray] = None,
        # --- geometry params ---
        num_lanes_total: int = 4,
        lane_width: float = 4.0,
        lane_directions: Optional[List[int]] = None,  # e.g., [+1,+1,-1,-1]; if None, first half +1, rest -1
        store_history: bool = True,
        # --- radio propagation params ---
        noise_mw: float = 1e-9,              # thermal noise in mW
        pathloss_n_los: float = 2.0,         # LOS path-loss exponent
        pathloss_n_nlos: float = 3.5,        # NLOS path-loss exponent
        desired_link_distance_m: float = 10.0,  # desired signal link distance in meters
        # --- init spacing ---
        min_initial_gap: float = 2.0         # meters; ≥2 m within each lane at t=0
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
        self.bler_table = None
        self._bler_points = None  # DataFrame with columns: resource, snr_db, bler
        
        self._init_bler_from_csv(bler_table_path)
       
        if bler_table_path is not None:
            self.bler_table= pd.read_csv(bler_table_path)
            df = self.bler_table
            cols = set(df.columns)
            points = []
            # If there are explicit points
            if {"resource", "snr_db", "bler"}.issubset(cols):
                points.append(df[["resource", "snr_db", "bler"]].copy())
            # If there are bins, convert to midpoints for interpolation
            if {"resource", "snr_min_db", "snr_max_db", "bler"}.issubset(cols):
                mid = 0.5 * (df["snr_min_db"] + df["snr_max_db"])
                p2 = pd.DataFrame({"resource": df["resource"], "snr_db": mid, "bler": df["bler"]})
                points.append(p2)
            if points:
                self._bler_points = pd.concat(points, ignore_index=True)
                self._bler_points.sort_values(["resource", "snr_db"], inplace=True)
        self.rng = np.random.default_rng(seed)
        self.R = 3  # actions/resources

        # Radio / propagation params
        self.noise_mw = noise_mw
        self.pathloss_n_los = pathloss_n_los
        self.pathloss_n_nlos = pathloss_n_nlos
        self.desired_link_distance_m = desired_link_distance_m

        # --- Lanes / directions ---
        assert num_lanes_total >= 1
        self.num_lanes_total = num_lanes_total
        self.lane_width = lane_width
        if lane_directions is None:
            half = num_lanes_total // 2
            lane_directions = [+1] * max(1, half) + [-1] * (num_lanes_total - max(1, half))
        assert len(lane_directions) == num_lanes_total
        assert all(d in (-1, +1) for d in lane_directions)
        self.lane_directions = np.array(lane_directions, dtype=int)
        self.lane_centers_y = np.arange(num_lanes_total, dtype=float) * lane_width

        # --- Initial positions/speeds/lanes with min gap ---
        self.x0 = np.full(self.N, np.nan)
        self.v0 = self.rng.uniform(self.v_min, self.v_max, size=self.N)

        # random lanes with mild balance
        self.lane0 = self.rng.integers(low=0, high=self.num_lanes_total, size=self.N)

        min_gap = min_initial_gap  # use configured gap
        for lane in range(self.num_lanes_total):
            idx = np.where(self.lane0 == lane)[0]
            if idx.size == 0:
                continue
            # equally spaced within lane
            gaps = self.L / idx.size
            positions = np.array([k * gaps for k in range(idx.size)], dtype=float)
            # add small jitter but keep min_gap
            jitter = self.rng.uniform(-0.2 * gaps, 0.2 * gaps, size=idx.size)
            positions = (positions + jitter) % self.L
            positions.sort()
            # enforce min_gap
            for j in range(1, len(positions)):
                if positions[j] - positions[j - 1] < min_gap:
                    positions[j] = positions[j - 1] + min_gap
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

        # History buffers
        self.store_history = store_history
        self._reset_history()

    # ---------- SQLite helpers ----------
    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _p_success_from_snr_bin(self, resource: int, snr_db: float) -> Optional[float]:
        """
        Look up P(success) from 'snr_success_prob' table by resource and SNR bin.
        Returns None if no matching bin is found or on DB error.
        Schema:
          snr_success_prob(resource INTEGER, snr_min_db REAL, snr_max_db REAL, p_success REAL)
        """

        # --- CSV mode: exact bin if present ---
        if self.bler_table is not None:
            df = self.bler_table
            cols = set(df.columns)
            if {"resource", "snr_min_db", "snr_max_db", "bler"}.issubset(cols):
                mask = (
                    (df["resource"] == resource) &
                    (snr_db >= df["snr_min_db"]) &
                    (snr_db <  df["snr_max_db"])
                )
                row = df[mask]
                if not row.empty:
                    bler = float(row["bler"].iloc[0])
                    return max(0.0, min(1.0, 1.0 - bler))
            # If no matching bin (or only point data), let interpolation attempt it.
            return None

        # Mode 2: database
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT p_success
                    FROM snr_success_prob
                    WHERE resource = ?
                      AND ? >= snr_min_db AND ? < snr_max_db
                    LIMIT 1
                    """,
                    (int(resource), float(snr_db), float(snr_db))
                )
                row = cur.fetchone()
                if row:
                    return float(row[0])
        except Exception:
            pass
        return None

    def _p_success_from_snr_interp(self, resource: int, snr_db: float) -> Optional[float]:
        """
        Linearly interpolate P(success) from 'snr_success_curve' points.
        Requires at least one point on each side to interpolate; otherwise uses nearest.
        Schema:
          snr_success_curve(resource INTEGER, snr_db REAL, p_success REAL)
        """

       # --- CSV mode: interpolate using points if available; else derive from bins ---
        if self.bler_table is not None:
            df = self.bler_table
            cols = set(df.columns)

            # Prefer prepared points if we built them
            if self._bler_points is not None:
                P = self._bler_points[self._bler_points["resource"] == resource]
                if not P.empty:
                    left  = P[P["snr_db"] <= snr_db].sort_values("snr_db", ascending=False).head(1)
                    right = P[P["snr_db"] >  snr_db].sort_values("snr_db", ascending=True ).head(1)
                    if not left.empty and not right.empty:
                        x0, y0 = float(left["snr_db"].iloc[0]),  1.0 - float(left["bler"].iloc[0])
                        x1, y1 = float(right["snr_db"].iloc[0]), 1.0 - float(right["bler"].iloc[0])
                        if x1 == x0:
                            return max(0.0, min(1.0, 0.5*(y0+y1)))
                        t = (snr_db - x0) / (x1 - x0)
                        return max(0.0, min(1.0, y0 + t*(y1 - y0)))
                    elif not left.empty:
                        return max(0.0, min(1.0, 1.0 - float(left["bler"].iloc[0])))
                    elif not right.empty:
                        return max(0.0, min(1.0, 1.0 - float(right["bler"].iloc[0])))

            # If we reach here and only have bins, interpolate using bin-edge midpoints on-the-fly
            if {"resource", "snr_min_db", "snr_max_db", "bler"}.issubset(cols):
                B = df[df["resource"] == resource].copy()
                if B.empty:
                    return None
                B = B.sort_values("snr_min_db")
                # Left neighbor: highest bin with snr_max_db <= snr_db
                left_bins  = B[B["snr_max_db"] <= snr_db]
                right_bins = B[B["snr_min_db"] >  snr_db]
                if not left_bins.empty and not right_bins.empty:
                    l = left_bins.iloc[-1]
                    r = right_bins.iloc[0]
                    x0 = 0.5*(float(l["snr_min_db"]) + float(l["snr_max_db"]))
                    y0 = 1.0 - float(l["bler"])
                    x1 = 0.5*(float(r["snr_min_db"]) + float(r["snr_max_db"]))
                    y1 = 1.0 - float(r["bler"])
                    if x1 == x0:
                        return max(0.0, min(1.0, 0.5*(y0+y1)))
                    t = (snr_db - x0) / (x1 - x0)
                    return max(0.0, min(1.0, y0 + t*(y1 - y0)))
                elif not left_bins.empty:
                    l = left_bins.iloc[-1]
                    return max(0.0, min(1.0, 1.0 - float(l["bler"])))
                elif not right_bins.empty:
                    r = right_bins.iloc[0]
                    return max(0.0, min(1.0, 1.0 - float(r["bler"])))
            return None

        #Mode 2 database    
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                # left <= snr_db
                cur.execute(
                    """
                    SELECT snr_db, p_success
                    FROM snr_success_curve
                    WHERE resource = ? AND snr_db <= ?
                    ORDER BY snr_db DESC LIMIT 1
                    """,
                    (int(resource), float(snr_db))
                )
                left = cur.fetchone()
                # right > snr_db
                cur.execute(
                    """
                    SELECT snr_db, p_success
                    FROM snr_success_curve
                    WHERE resource = ? AND snr_db > ?
                    ORDER BY snr_db ASC LIMIT 1
                    """,
                    (int(resource), float(snr_db))
                )
                right = cur.fetchone()
            if left and right:
                x0, y0 = float(left[0]), float(left[1])
                x1, y1 = float(right[0]), float(right[1])
                if x1 == x0:
                    return max(0.0, min(1.0, 0.5 * (y0 + y1)))
                t = (snr_db - x0) / (x1 - x0)
                return max(0.0, min(1.0, y0 + t * (y1 - y0)))
            elif left:
                return max(0.0, min(1.0, float(left[1])))
            elif right:
                return max(0.0, min(1.0, float(right[1])))
        except Exception:
            pass
        return None

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
        """P(NLOSv) = 1 - P_LOS."""
        return 1.0 - self.p_los_highway(distance_m)

    # ---------- Radio helpers ----------
    def compute_tx_power_mw(self, tx_power_dbm: float) -> float:
        """Convert transmit power from dBm to mW."""
        return 10.0 ** (tx_power_dbm / 10.0)

    def _path_loss_gain_sample(self, d_m: float, n_los: float, n_nlos: float) -> float:
        """
        Sample LOS/NLOS according to 3GPP highway model and return linear path-loss gain ~ 1/d^n.
        This is stochastic (uses self.rng).
        """
        d = max(1e-3, float(d_m))  # clamp very small distances
        if self.rng.random() < self.p_los_highway(d):
            n = n_los
        else:
            n = n_nlos
        return 1.0 / (d**n + 1e-12)

    def compute_sinr_linear(self, tx_idx: int, actions: np.ndarray) -> float:
        """
        SINR = S / (I + N) in linear units (mW/mW).
        - Desired received power S: uses a configurable desired-link distance with LOS/NLOS sampling.
        - Interference I: sum of all co-channel interferers within interference_max_range, each sampled LOS/NLOS.
        - Thermal noise N: self.noise_mw (mW).
        """
        ai = int(actions[tx_idx])
        y_now = self._lane_y(self.lane)
        xi, yi = float(self.x[tx_idx]), float(y_now[tx_idx])

        # Desired signal
        pt_dbm = float(self.tx_power_dbm[tx_idx])
        pt_mw = self.compute_tx_power_mw(pt_dbm)
        d_sig = float(self.desired_link_distance_m)
        g_sig = self._path_loss_gain_sample(d_sig, self.pathloss_n_los, self.pathloss_n_nlos)
        S = pt_mw * g_sig

        # Interference
        I = 0.0
        for j in range(self.N):
            if j == tx_idx or int(actions[j]) != ai:
                continue
            xj, yj = float(self.x[j]), float(y_now[j])
            d_ij = self._ring_distance_2d(xi, yi, xj, yj)
            if d_ij > self.interf_range:
                continue
            ptj_dbm = float(self.tx_power_dbm[j])
            ptj_mw = self.compute_tx_power_mw(ptj_dbm)
            g_ij = self._path_loss_gain_sample(d_ij, self.pathloss_n_los, self.pathloss_n_nlos)
            I += ptj_mw * g_ij

        N = float(self.noise_mw)
        return S / (I + N + 1e-18)

    def compute_sinr_db(self, tx_idx: int, actions: np.ndarray) -> float:
        sinr_lin = self.compute_sinr_linear(tx_idx, actions)
        return 10.0 * np.log10(max(1e-18, sinr_lin))

    # ---------- Helpers: geometry, distances, history ----------
    def _lane_y(self, lane_idx: np.ndarray) -> np.ndarray:
        return self.lane_centers_y[lane_idx]

    def _ring_dx(self, xi, xj) -> float:
        """Signed shortest dx on the ring (for completeness)."""
        dx = (xj - xi + self.L / 2) % self.L - self.L / 2
        return float(dx)

    def _ring_distance_2d(self, xi, yi, xj, yj) -> float:
        """Euclidean distance using ring metric in x and direct in y."""
        dx = abs((xi - xj + self.L / 2) % self.L - self.L / 2)
        dy = abs(yi - yj)
        return float(np.hypot(dx, dy))

    def _reset_history(self):
        self.hist = {
            "x": [],
            "lane": [],
            "y": [],
            "v": [],
            "labels": [],
            "actions": []
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

        # reproducible RNG
        unavailable = (self.rng.random(self.N) < self.p_block_unavailable)

        rewards = np.zeros(self.N, dtype=float)
        labels = np.zeros(self.N, dtype=int)

        for i in range(self.N):
            if unavailable[i]:
                labels[i] = 3  # unavailable
                rewards[i] = 0.0
                continue

            sinr_db = self.compute_sinr_db(i, actions)
            ai = int(actions[i])

            # 1) Try DB binning
            p_success = self._p_success_from_snr_bin(ai, sinr_db)
            # 2) Else try interpolation
            if p_success is None:
                p_success = self._p_success_from_snr_interp(ai, sinr_db)
            # 3) If still None, conservative default = 0
            if p_success is None:
                p_success = 0.0

            ok = (self.rng.random() < float(p_success))
            labels[i]  = 1 if ok else 2
            rewards[i] = 10.0 if ok else -5.0

        # Kinematics: update once per step
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

        for _ in range(self.T + 1):
            acts = np.zeros(self.N, dtype=int)
            for i in range(self.N):
                pi = policies[i][self.s[i]]
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
        """Basic geometry summary: lane centers, directions, lane counts at t=0."""
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
                diffs = np.diff(xs, append=xs[0] + L)  # ring gaps
                mean_gap[t, ell] = diffs.mean()
                min_gap[t, ell] = diffs.min()
        return {"mean_gap_per_lane": mean_gap, "min_gap_per_lane": min_gap}


# ---------------------------
# Thesis policies
# ---------------------------

def behavior_policy() -> np.ndarray:
    """π_b(a|s): uniform over actions for every label."""
    return np.ones((4, 3)) / 3.0

def evaluation_policy() -> np.ndarray:
    """π_e(a|s): biased towards action 0 on {idle, success}; exploratory on {collision, unavailable}."""
    p = np.zeros((4, 3))
    p[0] = np.array([0.7, 0.2, 0.1])  # idle
    p[1] = np.array([0.7, 0.2, 0.1])  # success
    p[2] = np.array([0.5, 0.3, 0.2])  # collision
    p[3] = np.array([0.5, 0.3, 0.2])  # unavailable
    return p

def _init_bler_from_csv(self, bler_table_path: Optional[str]):
    """Load CSV and build self.bler_table and self._bler_points (for interpolation)."""
    self.bler_table = None
    self._bler_points = None
    if bler_table_path is None:
        return

    df = pd.read_csv(bler_table_path)
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Map your file's columns to standard names
    if "sinr_db" in df.columns and "mcs_0_bler" in df.columns:
        df = df.rename(columns={"sinr_db": "snr_db", "mcs_0_bler": "bler"})
        df["resource"] = 0   # assume single resource
    else:
        raise ValueError("CSV does not contain expected columns: 'sinr_db' and 'mcs_0_bler'")

    self.bler_table = df.copy()

    # Build points table for interpolation
    self._bler_points = df[["resource", "snr_db", "bler"]].copy()
    self._bler_points.sort_values(["resource", "snr_db"], inplace=True)




# ---------------------------
# Convenience runner
# ---------------------------

def simulate_highway_multiagent_sql(
    N: int = 20, T: int = 100, seed: int = 20,
    db_path: str = None,
    bler_table_path: Optional[str] = None,
    interference_max_range: float = 500.0,
    num_lanes_total: int = 4,
    lane_width: float = 3.75,
    lane_directions: Optional[List[int]] = None,
    noise_mw: float = 1e-9,
    p_block_unavailable: float = 0.0,  
    store_history: bool = True,
    tx_power_dbm: Optional[np.ndarray] = None,  
    pathloss_n_los: float = 2.0,         # LOS path-loss exponent
    pathloss_n_nlos: float = 3.5,        # NLOS path-loss exponent
    desired_link_distance_m: float = 10.0,  # desired signal link distance in meters
    min_initial_gap: float = 2.0         # meters; ≥2 m within each lane at t=0
) -> dict:
    """
    Runs one rollout with behavior policy and one with evaluation policy.
    Returns time series of avg rewards and label rates, raw histories, and geometry summaries.
    """
    env = HighwayMultiAgentEnvSQL(
        num_agents=N, T=T, seed=seed, db_path=db_path, bler_table_path=bler_table_path, 
        interference_max_range=interference_max_range,
        num_lanes_total=num_lanes_total, lane_width=lane_width, noise_mw=noise_mw,
        p_block_unavailable=p_block_unavailable, tx_power_dbm=tx_power_dbm,
        pathloss_n_los=pathloss_n_los, pathloss_n_nlos=pathloss_n_nlos,
        desired_link_distance_m=desired_link_distance_m,
        min_initial_gap=min_initial_gap,   
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
