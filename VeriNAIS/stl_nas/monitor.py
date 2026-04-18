"""
stl_nas/monitor.py
------------------
Online STL monitor backed by RTAMT (https://github.com/nickovic/rtamt).

The monitor receives the growing signal prefix σ[0 : t] at each NAS
iteration and evaluates all eight STL properties incrementally.
RTAMT uses a sliding-window algorithm with amortised O(1) cost per time
step per property (Proposition 1 of the paper).

Monitoring overhead
-------------------
  Per-candidate: O(k · d) where k=8 properties, d=7 signal dimensions
  Total:          O(k · d · N · T) — linear in search budget N×T
  Empirically:    < 1 ms per candidate on the evaluation machine.

Design note
-----------
The gatekeeper calls this monitor twice per candidate:
  1. push_hypothetical(σ_test) — evaluate the hypothetical signal extension
  2. pop_hypothetical()        — roll back if the candidate is rejected
This avoids copying the entire signal history for each candidate.

Auxiliary signals
-----------------
Two properties require auxiliary time-series derived from the raw signal:

  s_acc_lag3_diff(t) = s_acc(t) − s_acc(t−3)
      Used by φ_monotone.  For t < 3, the lag is replaced by s_acc(0).

  is_improving(t) = 1 if s_acc(t) > s_acc(t−3) − ε, else 0
      Used by φ_stagnation as the antecedent predicate.

Both are computed by the monitor and fed to RTAMT as additional input
channels alongside the raw 7D signal.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .signal import SearchSignal, SignalHistory
from .properties import STLPropertyLibrary, STLProperty


# ---------------------------------------------------------------------------
# RTAMT availability guard
# ---------------------------------------------------------------------------

try:
    import rtamt                     # type: ignore
    _RTAMT_AVAILABLE = True
except ImportError:
    _RTAMT_AVAILABLE = False


# ---------------------------------------------------------------------------
# STLMonitor
# ---------------------------------------------------------------------------

class STLMonitor:
    """Incrementally evaluates all eight STL properties on the search signal.

    Parameters
    ----------
    library :
        STLPropertyLibrary instance (provides specifications and thresholds).
    rho_threshold :
        Minimum robustness required for a candidate to pass (ρ_th in the
        paper; default 0 enforces hard constraint satisfaction).
    use_rtamt :
        If True and RTAMT is installed, use the RTAMT backend for online
        evaluation.  If False (or RTAMT is unavailable), falls back to the
        pure-NumPy robustness implementation in STLPropertyLibrary.

    Attributes
    ----------
    rho_min_history : list[float]
        Records the minimum robustness across all properties at each committed
        iteration.  Useful for post-hoc trajectory analysis.
    """

    def __init__(
        self,
        library: STLPropertyLibrary,
        rho_threshold: float = 0.0,
        use_rtamt: bool = True,
    ) -> None:
        self.library = library
        self.rho_threshold = rho_threshold
        self._use_rtamt = use_rtamt and _RTAMT_AVAILABLE

        self._history = SignalHistory()
        self._hypothetical_pushed = False

        # Per-property robustness at each committed step
        self.rho_history: List[Dict[str, float]] = []
        self.rho_min_history: List[float] = []

        if self._use_rtamt:
            self._monitors: Dict[str, "rtamt.StlOnlineMonitor"] = {}
            self._build_rtamt_monitors()

    # ------------------------------------------------------------------ #
    #  RTAMT monitor construction                                          #
    # ------------------------------------------------------------------ #

    def _build_rtamt_monitors(self) -> None:
        """Compile one RTAMT online monitor per property."""
        for prop in self.library.properties:
            mon = rtamt.StlOnlineMonitor()
            # Declare all signal variables used by this property
            for dim in prop.signal_dims:
                mon.declare_var(dim, "float")
            # Declare auxiliary variables for properties that need them
            if prop.name == "phi_monotone":
                mon.declare_var("s_acc_lag3_diff", "float")
            if prop.name == "phi_stagnation":
                mon.declare_var("is_improving", "float")
            mon.spec = prop.rtamt_spec
            mon.parse()
            self._monitors[prop.name] = mon

    # ------------------------------------------------------------------ #
    #  Signal accessors                                                    #
    # ------------------------------------------------------------------ #

    @property
    def t(self) -> int:
        """Current committed iteration index (0-based)."""
        return len(self._history)

    def _make_auxiliary(self, sig: SearchSignal) -> Dict[str, float]:
        """Compute auxiliary signal values for a candidate signal step.

        Returns a dict with keys 's_acc_lag3_diff' and 'is_improving',
        derived from the current history plus this new step.
        """
        th = self.library.thresholds
        eps = th["epsilon"]
        w   = int(th["window"])

        # s_acc(t−3): use history if available, else s_acc(0)
        if self.t >= w:
            acc_lag = self._history[self.t - w].s_acc
        elif self.t > 0:
            acc_lag = self._history[0].s_acc
        else:
            acc_lag = sig.s_acc   # t=0: no lag available, set to current

        lag_diff   = sig.s_acc - acc_lag
        is_improv  = 1.0 if lag_diff > -eps else 0.0

        return {
            "s_acc_lag3_diff": lag_diff,
            "is_improving":    is_improv,
        }

    def _sig_to_rtamt_inputs(
        self,
        sig: SearchSignal,
        aux: Dict[str, float],
        t: int,
    ) -> Dict[str, list]:
        """Convert a signal step to the dict-of-lists format expected by RTAMT."""
        raw = {
            "s_acc":    sig.s_acc,
            "s_energy": sig.s_energy,
            "s_params": sig.s_params,
            "s_conv":   sig.s_conv,
            "s_hv":     sig.s_hv,
            "s_lat":    sig.s_lat,
            "s_pwr":    sig.s_pwr,
        }
        raw.update(aux)
        # RTAMT expects: {var_name: [(timestamp, value), ...]}
        return {k: [(t, v)] for k, v in raw.items()}

    # ------------------------------------------------------------------ #
    #  Core evaluation interface                                           #
    # ------------------------------------------------------------------ #

    def evaluate_hypothetical(
        self,
        candidate_sig: SearchSignal,
    ) -> Dict[str, float]:
        """Evaluate all properties on σ[0:t] ++ candidate_sig.

        Pushes the candidate step onto the signal history temporarily.
        Call pop_hypothetical() to roll back regardless of the outcome.

        Parameters
        ----------
        candidate_sig :
            The hypothetical signal extension σ_test = ComputeSignal(P_{t-1} ∪ {A}).

        Returns
        -------
        dict
            {property_name: robustness} for all 8 properties at the
            hypothetical extension.
        """
        self._history.append(candidate_sig)
        self._hypothetical_pushed = True

        t_hyp = len(self._history) - 1
        aux   = self._make_auxiliary(candidate_sig)

        if self._use_rtamt:
            rho = self._evaluate_rtamt(candidate_sig, aux, t_hyp)
        else:
            rho = self._evaluate_numpy()

        return rho

    def pop_hypothetical(self) -> None:
        """Roll back the last hypothetical signal step (candidate rejected)."""
        if self._hypothetical_pushed:
            self._history.pop()
            self._hypothetical_pushed = False

    def commit(self, sig: SearchSignal) -> Dict[str, float]:
        """Commit a signal step to the permanent history (candidate accepted).

        Should be called after evaluate_hypothetical returned a passing
        robustness, but BEFORE pop_hypothetical (i.e. the push is made
        permanent).

        Returns the same robustness dict as evaluate_hypothetical.
        """
        if not self._hypothetical_pushed:
            # No hypothetical was pushed; push and evaluate fresh
            self._history.append(sig)
            t_cur = len(self._history) - 1
            aux   = self._make_auxiliary(sig)
            if self._use_rtamt:
                rho = self._evaluate_rtamt(sig, aux, t_cur)
            else:
                rho = self._evaluate_numpy()
        else:
            # The hypothetical step is already at the top of history;
            # compute robustness (same as in evaluate_hypothetical) and
            # record it as committed.
            t_cur = len(self._history) - 1
            aux   = self._make_auxiliary(sig)
            if self._use_rtamt:
                rho = self._evaluate_rtamt(sig, aux, t_cur)
            else:
                rho = self._evaluate_numpy()
            self._hypothetical_pushed = False  # no longer hypothetical

        rho_min = min(rho.values())
        self.rho_history.append(rho)
        self.rho_min_history.append(rho_min)
        return rho

    # ------------------------------------------------------------------ #
    #  Backend implementations                                             #
    # ------------------------------------------------------------------ #

    def _evaluate_rtamt(
        self,
        sig: SearchSignal,
        aux: Dict[str, float],
        t: int,
    ) -> Dict[str, float]:
        """Evaluate all properties incrementally via RTAMT."""
        inputs = self._sig_to_rtamt_inputs(sig, aux, t)
        results: Dict[str, float] = {}

        for prop in self.library.properties:
            mon = self._monitors[prop.name]
            # Feed only the dimensions declared for this monitor
            relevant = {k: v for k, v in inputs.items()
                        if k in mon.declared_vars}
            rho_val = mon.update(relevant)
            # RTAMT returns None until enough history is available; treat as +∞
            results[prop.name] = float(rho_val) if rho_val is not None else np.inf

        return results

    def _evaluate_numpy(self) -> Dict[str, float]:
        """Evaluate all properties via the pure-NumPy backend."""
        H = self._history.to_array()   # shape (T, 7)
        return self.library.compute_robustness_all(H)

    # ------------------------------------------------------------------ #
    #  Gating decision                                                     #
    # ------------------------------------------------------------------ #

    def gate(self, rho: Dict[str, float]) -> bool:
        """Return True iff all properties meet the robustness threshold.

        This is the Accept/Reject decision in Algorithm 1, line 9 of the paper:
            ρ_min = min_i ρ(φ_i, [σ[0:t−1], σ_test], 0)  ≥  ρ_th
        """
        return all(v >= self.rho_threshold for v in rho.values())

    def gate_with_detail(
        self, rho: Dict[str, float]
    ) -> tuple[bool, Optional[str]]:
        """Gate decision with the name of the first violating property."""
        for name, val in rho.items():
            if val < self.rho_threshold:
                return False, name
        return True, None

    # ------------------------------------------------------------------ #
    #  Statistics                                                          #
    # ------------------------------------------------------------------ #

    def violation_summary(self) -> Dict[str, int]:
        """Count committed iterations where each property was violated."""
        counts: Dict[str, int] = {p.name: 0 for p in self.library.properties}
        for rho in self.rho_history:
            for name, val in rho.items():
                if val < self.rho_threshold:
                    counts[name] += 1
        return counts

    def overall_satisfaction_rate(self) -> float:
        """Fraction of committed iterations where ALL properties were satisfied."""
        if not self.rho_history:
            return 1.0
        sat = sum(
            1 for rho in self.rho_history
            if all(v >= self.rho_threshold for v in rho.values())
        )
        return sat / len(self.rho_history)
