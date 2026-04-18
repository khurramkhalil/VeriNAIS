"""
stl_nas/properties.py
---------------------
Eight STL properties constituting the VeriNAIS property library (Section IV).

Each property is defined as an RTAMT specification string together with its
signal dimension(s) and threshold parameters.  The STLPropertyLibrary class
collects all eight and exposes them to the STLMonitor.

Property summary (Table I of the paper)
----------------------------------------
Class                    Name              Formula
─────────────────────────────────────────────────────────────────────────────
Resource bounds          φ_energy          G(s_energy ≤ β_max)
                         φ_size            G(s_params ≤ γ_max_ratio)
                         φ_latency         G(s_lat    ≤ λ_max)          [NEW]
                         φ_power           G(s_pwr    ≤ P_max)          [NEW]
─────────────────────────────────────────────────────────────────────────────
Search dynamics          φ_monotone        G(s_acc(t) ≥ s_acc(t-3) − ε)
                         φ_diversity       G(s_conv   ≥ δ_min)
─────────────────────────────────────────────────────────────────────────────
Temporal interaction     φ_recovery        G(s_energy > 0.9·β_max →
                         [NOVEL]              F[1,3](s_energy < 0.7·β_max))
                         φ_stagnation      G(¬imp₃(t) →
                         [NOVEL]              s_params ≤ 0.8·γ_max_ratio)
─────────────────────────────────────────────────────────────────────────────

Default thresholds (all validated on NAS-Bench-201 / NATS-Bench TSS)
----------------------------------------------------------------------
β_max       = 1.0        energy proxy budget (raw ∈ [0,2]; 45 % exceed)
γ_max_ratio = 1.0        params / γ_max ratio budget (γ_max = 500 000)
λ_max       = 5.0 ms     latency budget (conv-heavy ≈ 6.7 ms → violates)
P_max       = 27.0 W     power budget   (conv-heavy ≈ 28.2 W → violates)
ε           = 0.01       accuracy monotonicity tolerance (%)
δ_min       = 0.2        minimum conv-op fraction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Default threshold values (calibrated on Jetson Thor measurements, §VII-A)
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: Dict[str, float] = {
    # Resource-bound properties
    "beta_max":       1.0,         # energy proxy budget
    "gamma_max_ratio": 1.0,        # params / γ_max (γ_max = 500 000 raw params)
    "lambda_max":     5.0,         # latency budget [ms]
    "P_max":          27.0,        # power budget [W]
    # Search-dynamics properties
    "epsilon":        0.01,        # accuracy monotonicity slack (%)
    "delta_min":      0.20,        # minimum conv-op diversity
    # Temporal thresholds (derived from β_max)
    "recovery_hi":    0.90,        # φ_recovery trigger level (× β_max)
    "recovery_lo":    0.70,        # φ_recovery target level  (× β_max)
    "stagnation_sz":  0.80,        # φ_stagnation size cap    (× γ_max_ratio)
    # Temporal window
    "window":         3,           # look-back / look-ahead steps
}


# ---------------------------------------------------------------------------
# Property descriptor
# ---------------------------------------------------------------------------

@dataclass
class STLProperty:
    """Metadata for one STL property.

    Attributes
    ----------
    name : str
        Short identifier (e.g. 'phi_energy').
    description : str
        Human-readable description of what the property enforces.
    rtamt_spec : str
        RTAMT specification string.  Signal variable names match the 7D
        signal dimensions in SearchSignal.DIMS.
    signal_dims : list[str]
        The subset of SearchSignal dimensions referenced by this property.
    lipschitz_constant : float
        Upper bound on the Lipschitz constant of the robustness function
        with respect to signal perturbations (used in Theorem 2).
    is_temporal_interaction : bool
        True for φ_recovery and φ_stagnation — the paper's novel properties
        that require genuine temporal operators.
    """
    name: str
    description: str
    rtamt_spec: str
    signal_dims: List[str]
    lipschitz_constant: float = 1.0
    is_temporal_interaction: bool = False


# ---------------------------------------------------------------------------
# Property library
# ---------------------------------------------------------------------------

class STLPropertyLibrary:
    """Collects and parametrises the eight STL properties.

    Parameters
    ----------
    thresholds :
        Override any subset of DEFAULT_THRESHOLDS.  Keys must match the
        DEFAULT_THRESHOLDS dict above.

    Usage
    -----
    >>> lib = STLPropertyLibrary()
    >>> for prop in lib.properties:
    ...     print(prop.name, prop.rtamt_spec)
    """

    def __init__(self, thresholds: Optional[Dict[str, float]] = None) -> None:
        self.thresholds: Dict[str, float] = dict(DEFAULT_THRESHOLDS)
        if thresholds:
            unknown = set(thresholds) - set(DEFAULT_THRESHOLDS)
            if unknown:
                raise ValueError(f"Unknown threshold keys: {unknown}")
            self.thresholds.update(thresholds)

        self.properties: List[STLProperty] = self._build_properties()

    # ------------------------------------------------------------------ #
    #  Property construction                                               #
    # ------------------------------------------------------------------ #

    def _build_properties(self) -> List[STLProperty]:
        th = self.thresholds
        props: List[STLProperty] = []

        # ── 1. φ_energy ──────────────────────────────────────────────────
        # G(s_energy(t) ≤ β_max)
        # Robustness: min_{t} (β_max − s_energy(t))
        # Lipschitz = 1 (standard G-type, Eq. 17 in paper)
        props.append(STLProperty(
            name="phi_energy",
            description=(
                f"Mean FLOP-normalised energy proxy must stay below "
                f"β_max={th['beta_max']:.2f} at every iteration."
            ),
            rtamt_spec=(
                f"always (s_energy <= {th['beta_max']})"
            ),
            signal_dims=["s_energy"],
            lipschitz_constant=1.0,
        ))

        # ── 2. φ_size ────────────────────────────────────────────────────
        # G(s_params(t) ≤ γ_max_ratio)
        # s_params = mean_params / γ_max → budget is γ_max_ratio = 1.0
        props.append(STLProperty(
            name="phi_size",
            description=(
                f"Mean parameter count (normalised by γ_max=500 000) must "
                f"stay below {th['gamma_max_ratio']:.2f} at every iteration."
            ),
            rtamt_spec=(
                f"always (s_params <= {th['gamma_max_ratio']})"
            ),
            signal_dims=["s_params"],
            lipschitz_constant=1.0,
        ))

        # ── 3. φ_latency  [NEW — Jetson Thor measured] ───────────────────
        # G(s_lat(t) ≤ λ_max)
        props.append(STLProperty(
            name="phi_latency",
            description=(
                f"Max population latency (Jetson Thor FP16) must stay below "
                f"λ_max={th['lambda_max']:.1f} ms at every iteration."
            ),
            rtamt_spec=(
                f"always (s_lat <= {th['lambda_max']})"
            ),
            signal_dims=["s_lat"],
            lipschitz_constant=1.0,
        ))

        # ── 4. φ_power  [NEW — Jetson Thor tegrastats] ───────────────────
        # G(s_pwr(t) ≤ P_max)
        props.append(STLProperty(
            name="phi_power",
            description=(
                f"Max population power draw (Jetson Thor) must stay below "
                f"P_max={th['P_max']:.1f} W at every iteration."
            ),
            rtamt_spec=(
                f"always (s_pwr <= {th['P_max']})"
            ),
            signal_dims=["s_pwr"],
            lipschitz_constant=1.0,
        ))

        # ── 5. φ_monotone ────────────────────────────────────────────────
        # G(s_acc(t) ≥ s_acc(t−3) − ε)
        # Implemented via a 3-step lag comparison in the monitor.
        # Lipschitz = 1 (the lag comparison is still a min-type operation).
        eps = th["epsilon"]
        w   = int(th["window"])
        props.append(STLProperty(
            name="phi_monotone",
            description=(
                f"Mean population accuracy must not drop more than ε={eps} "
                f"over any {w}-iteration window (prevents degenerative dips)."
            ),
            rtamt_spec=(
                # RTAMT does not support native lag; the monitor handles this
                # by computing the lag difference as an auxiliary signal.
                # Specification expressed in terms of the auxiliary signal
                # 's_acc_lag3_diff' = s_acc(t) - s_acc(t-3):
                f"always (s_acc_lag3_diff >= {-eps})"
            ),
            signal_dims=["s_acc"],
            lipschitz_constant=1.0,
        ))

        # ── 6. φ_diversity ───────────────────────────────────────────────
        # G(s_conv(t) ≥ δ_min)
        dmin = th["delta_min"]
        props.append(STLProperty(
            name="phi_diversity",
            description=(
                f"At least δ_min={dmin:.0%} of population edges must be "
                f"convolutional operations (prevents op-count collapse)."
            ),
            rtamt_spec=(
                f"always (s_conv >= {dmin})"
            ),
            signal_dims=["s_conv"],
            lipschitz_constant=1.0,
        ))

        # ── 7. φ_recovery  [NOVEL temporal interaction] ──────────────────
        # G(s_energy(t) > 0.9·β_max  →  F[1,3](s_energy(t') < 0.7·β_max))
        #
        # Robustness (Eq. 18):
        #   ρ = min_{t: s_e > 0.9β}  max_{t' ∈ [t+1,t+3]} (0.7β − s_e(t'))
        #
        # Lipschitz ≤ 1/0.7 ≈ 1.43 (standard STL composition result).
        hi  = th["recovery_hi"] * th["beta_max"]
        lo  = th["recovery_lo"] * th["beta_max"]
        props.append(STLProperty(
            name="phi_recovery",
            description=(
                "TEMPORAL INTERACTION: If mean energy exceeds 90 % of β_max "
                "at any iteration, within 3 iterations it must recover below "
                "70 % of β_max.  Prevents sustained high-energy exploration."
            ),
            rtamt_spec=(
                # Implication: p → q  encoded as  (¬p) ∨ q
                # p: s_energy > hi  (trigger)
                # q: F[1,3](s_energy < lo)  (bounded recovery)
                f"always ((s_energy <= {hi}) or "
                f"(eventually[1,3](s_energy < {lo})))"
            ),
            signal_dims=["s_energy"],
            lipschitz_constant=1.0 / th["recovery_lo"],   # ≈ 1.43
            is_temporal_interaction=True,
        ))

        # ── 8. φ_stagnation  [NOVEL temporal interaction] ────────────────
        # G(¬imp₃(t)  →  s_params(t) ≤ 0.8·γ_max_ratio)
        #
        # imp₃(t) ≡ s_acc(t) > s_acc(t−3) − ε   (improving over 3 iters)
        # ¬imp₃(t) ≡ s_acc(t) ≤ s_acc(t−3) − ε  (stagnating)
        #
        # When the search stagnates, the population must remain below 80 %
        # of the parameter budget (≈ 400 K params, the NATSBench mean).
        # Prevents wasteful high-complexity exploration during stagnation.
        #
        # Implemented via auxiliary signal 'stagnating': 1 if ¬imp₃, else 0.
        sz_cap = th["stagnation_sz"] * th["gamma_max_ratio"]
        props.append(STLProperty(
            name="phi_stagnation",
            description=(
                "TEMPORAL INTERACTION: If accuracy has not improved over the "
                "last 3 iterations (stagnation), the population must stay "
                "below 80 % of the parameter budget."
            ),
            rtamt_spec=(
                # Implication: ¬imp₃ → s_params ≤ cap
                # Encoded as: imp₃ ∨ (s_params ≤ cap)
                # The auxiliary signal 'is_improving' is 1 when imp₃ holds.
                f"always ((is_improving >= 1.0) or "
                f"(s_params <= {sz_cap:.6f}))"
            ),
            signal_dims=["s_acc", "s_params"],
            lipschitz_constant=1.0 / th["stagnation_sz"],   # ≈ 1.25
            is_temporal_interaction=True,
        ))

        return props

    # ------------------------------------------------------------------ #
    #  Convenience accessors                                               #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.properties)

    def __iter__(self):
        return iter(self.properties)

    def __getitem__(self, name: str) -> STLProperty:
        for p in self.properties:
            if p.name == name:
                return p
        raise KeyError(name)

    @property
    def names(self) -> List[str]:
        return [p.name for p in self.properties]

    @property
    def temporal_properties(self) -> List[STLProperty]:
        """Return only φ_recovery and φ_stagnation."""
        return [p for p in self.properties if p.is_temporal_interaction]

    @property
    def resource_properties(self) -> List[STLProperty]:
        """Return the four resource-bound properties."""
        return [p for p in self.properties
                if p.name in ("phi_energy", "phi_size",
                               "phi_latency", "phi_power")]

    # ------------------------------------------------------------------ #
    #  Robustness computation (pure Python, no RTAMT dependency)           #
    # ------------------------------------------------------------------ #

    def compute_robustness_all(
        self,
        signal_history: "np.ndarray",   # shape (T, 7)
        thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Evaluate all eight properties on a complete signal trajectory.

        This pure-NumPy implementation is used for post-hoc analysis and
        unit-testing.  The STLMonitor (monitor.py) uses RTAMT for online
        evaluation during search.

        Parameters
        ----------
        signal_history :
            Array of shape (T, 7) where rows are iterations and columns are
            the 7 signal dimensions in SearchSignal.DIMS order.
        thresholds :
            Optional threshold overrides (same keys as DEFAULT_THRESHOLDS).

        Returns
        -------
        dict
            {property_name: robustness_value} for all 8 properties.
        """
        th = dict(self.thresholds)
        if thresholds:
            th.update(thresholds)

        H = np.asarray(signal_history, dtype=np.float64)
        if H.ndim != 2 or H.shape[1] != 7:
            raise ValueError(f"Expected shape (T, 7), got {H.shape}")

        T = H.shape[0]
        # Unpack columns
        s_acc    = H[:, 0]
        s_energy = H[:, 1]
        s_params = H[:, 2]
        s_conv   = H[:, 3]
        # s_hv   = H[:, 4]  -- diagnostic only, no property
        s_lat    = H[:, 5]
        s_pwr    = H[:, 6]

        results: Dict[str, float] = {}

        # ── Resource bounds (G-type, robustness = min_t (threshold − signal)) ──
        results["phi_energy"] = float(
            np.min(th["beta_max"] - s_energy)
        )
        results["phi_size"] = float(
            np.min(th["gamma_max_ratio"] - s_params)
        )
        results["phi_latency"] = float(
            np.min(th["lambda_max"] - s_lat)
        )
        results["phi_power"] = float(
            np.min(th["P_max"] - s_pwr)
        )

        # ── φ_monotone ────────────────────────────────────────────────────────
        # ρ = min_t  (s_acc(t) − s_acc(t−3) + ε)
        # For t < 3, compare against s_acc(0) (see §V-B of the paper).
        eps = th["epsilon"]
        w   = int(th["window"])
        lag = np.concatenate([
            np.full(w, s_acc[0]),   # pad the first w steps with s_acc(0)
            s_acc[:-w],
        ])
        results["phi_monotone"] = float(
            np.min(s_acc - lag + eps)
        )

        # ── φ_diversity ───────────────────────────────────────────────────────
        results["phi_diversity"] = float(
            np.min(s_conv - th["delta_min"])
        )

        # ── φ_recovery  (Eq. 18) ─────────────────────────────────────────────
        # ρ = min_{t: s_e > hi}  max_{t' ∈ [t+1, t+3]} (lo − s_e(t'))
        hi = th["recovery_hi"] * th["beta_max"]
        lo = th["recovery_lo"] * th["beta_max"]

        trigger_times = np.where(s_energy > hi)[0]
        if len(trigger_times) == 0:
            # Property vacuously satisfied — no trigger occurred
            results["phi_recovery"] = float(lo - np.min(s_energy))
        else:
            worst = np.inf
            for t in trigger_times:
                # Look-ahead window [t+1, t+3] clamped to trajectory length
                lo_idx = min(t + 1, T - 1)
                hi_idx = min(t + w, T - 1)
                if lo_idx > hi_idx:
                    # No future steps available — conservatively set to -∞
                    worst = -np.inf
                    break
                window_vals = s_energy[lo_idx: hi_idx + 1]
                # Recovery satisfied if min energy in window is < lo
                recovery_margin = lo - np.min(window_vals)
                worst = min(worst, recovery_margin)
            results["phi_recovery"] = float(worst)

        # ── φ_stagnation ─────────────────────────────────────────────────────
        # ρ = min_t  max(is_improving(t) − 0.5,  sz_cap − s_params(t))
        # is_improving(t) = 1 if s_acc(t) > s_acc(t−3) − ε, else 0
        sz_cap = th["stagnation_sz"] * th["gamma_max_ratio"]
        is_improving = (s_acc - lag + eps > 0).astype(float)

        # Implication robustness: ρ(p→q) = max(−ρ(p), ρ(q))
        # p: ¬imp₃  →  ρ(p) = 0.5 − is_improving  (1 if not improving, -1 if improving)
        # q: s_params ≤ sz_cap
        rho_p = 0.5 - is_improving          # positive when NOT improving
        rho_q = sz_cap - s_params
        rho_impl = np.maximum(-rho_p, rho_q)
        results["phi_stagnation"] = float(np.min(rho_impl))

        return results

    def is_satisfied(
        self,
        signal_history: "np.ndarray",
        rho_threshold: float = 0.0,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, bool]:
        """Return True/False satisfaction for each property."""
        rho = self.compute_robustness_all(signal_history, thresholds)
        return {name: rho[name] >= rho_threshold for name in rho}
