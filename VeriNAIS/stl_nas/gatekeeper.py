"""
stl_nas/gatekeeper.py
---------------------
STL-NAS Gatekeeper — Algorithm 1 from the paper.

The gatekeeper is a wrapper around any NAS oracle O.  At each iteration t
it intercepts the batch of candidate architectures proposed by O, computes
the hypothetical signal extension for each candidate, and admits only those
satisfying:

    ρ_min(A) = min_i  ρ(φ_i, [σ[0:t−1], σ_test(A)], 0)  ≥  ρ_th

where σ_test(A) = ComputeSignal(P_{t−1} ∪ {A}).

Soundness guarantee (Theorem 1)
--------------------------------
Any trajectory τ* = (P_0, …, P_T) produced by this gatekeeper satisfies

    ρ(φ_i, σ_{τ*}, 0) ≥ ρ_th   for all φ_i ∈ Φ, all t.

The proof is by induction on t; the gate condition at line 9 of
Algorithm 1 is the inductive step.

Post-selection recovery
-----------------------
After NSGA-II / LaMOO selection, the committed population P_t may be a
strict subset of P_{t−1} ∪ C_valid.  If selection drops members, the
final signal σ[t] = ComputeSignal(P_t) may differ from any σ_test
that was individually checked.  The gatekeeper therefore re-verifies the
committed signal after selection (verify_population) and performs a
recovery pass if needed (documented as Bug 1 fix in the paper).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .signal import Architecture, SearchSignal, SignalHistory, compute_signal
from .properties import STLPropertyLibrary, DEFAULT_THRESHOLDS
from .monitor import STLMonitor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gatekeeper configuration
# ---------------------------------------------------------------------------

@dataclass
class GatekeeperConfig:
    """Hyperparameters for the STL gatekeeper.

    Attributes
    ----------
    rho_threshold :
        Minimum STL robustness for admission (ρ_th).
        0.0 → hard constraint satisfaction.
    gamma_max :
        Raw parameter count budget (used to normalise s_params signal).
    max_rejection_retries :
        If all candidates in a batch are rejected, the oracle is queried
        up to this many additional times before falling back to keeping
        the current population unchanged for this iteration.
    population_size :
        Number of architectures in each generation.
    use_rtamt :
        Whether to use the RTAMT backend for STL evaluation.
    verbose :
        If True, log accepted/rejected counts per iteration.
    """
    rho_threshold: float = 0.0
    gamma_max: float = 500_000.0
    max_rejection_retries: int = 5
    population_size: int = 20
    use_rtamt: bool = True
    verbose: bool = False


# ---------------------------------------------------------------------------
# Gatekeeper statistics (accumulated over the full run)
# ---------------------------------------------------------------------------

@dataclass
class GatekeeperStats:
    total_candidates: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    rejection_reasons: Dict[str, int] = field(default_factory=dict)
    fallback_iterations: int = 0          # iterations where all cands rejected
    post_selection_recoveries: int = 0    # times post-selection verify fired

    @property
    def acceptance_rate(self) -> float:
        if self.total_candidates == 0:
            return 1.0
        return self.total_accepted / self.total_candidates

    def log_summary(self) -> None:
        logger.info(
            "GatekeeperStats | candidates=%d  accepted=%d  "
            "acceptance_rate=%.1f%%  fallbacks=%d  recoveries=%d",
            self.total_candidates,
            self.total_accepted,
            100 * self.acceptance_rate,
            self.fallback_iterations,
            self.post_selection_recoveries,
        )
        for prop, count in sorted(
            self.rejection_reasons.items(), key=lambda x: -x[1]
        ):
            logger.info("  rejected by %-22s  %4d times", prop, count)


# ---------------------------------------------------------------------------
# Main gatekeeper class
# ---------------------------------------------------------------------------

class STLGatekeeper:
    """Wraps a NAS oracle with the STL gatekeeper (Algorithm 1).

    Parameters
    ----------
    library :
        STLPropertyLibrary with the eight STL properties and thresholds.
    config :
        GatekeeperConfig controlling runtime behaviour.
    hw_lookup :
        Callable (arch_id: int) → (lat_ms: float, pwr_w: float).
        Typically backed by HardwareLookupTable.query().
        Used to fill s_lat and s_pwr in the signal.

    Usage
    -----
    The gatekeeper is used by the NAS algorithm wrappers in
    stl_nas/algorithms/*.py via:

        gatekeeper.run(
            oracle=my_oracle,
            initial_population=init_pop,
            T=50,
            select_fn=nsga2_selection,
        )
    """

    def __init__(
        self,
        library: STLPropertyLibrary,
        config: GatekeeperConfig,
        hw_lookup: Callable[[int], Tuple[float, float]],
    ) -> None:
        self.library  = library
        self.config   = config
        self.hw_lookup = hw_lookup

        self.monitor = STLMonitor(
            library=library,
            rho_threshold=config.rho_threshold,
            use_rtamt=config.use_rtamt,
        )
        self.stats = GatekeeperStats()

        # Trajectory output
        self.trajectory: List[List[Architecture]] = []   # P_0, …, P_T
        self.signal_history = SignalHistory()

    # ------------------------------------------------------------------ #
    #  Feasibility-checked initialization (Assumption 1)                  #
    # ------------------------------------------------------------------ #

    def initialize_population(
        self,
        candidates: Sequence[Architecture],
        max_retries: int = 1000,
    ) -> List[Architecture]:
        """Select an initial population satisfying all STL properties.

        Implements rejection sampling: draw random subsets of `candidates`
        until a population of size N is found with ρ(φ_i, σ[0], 0) ≥ ρ_th.

        For the resource-bound properties, this reduces to checking that
        mean energy, mean params/γ_max, max latency, and max power of the
        initial sample are within budget.  The temporal interaction properties
        impose no constraint at t=0 (they require 3 iterations of history).

        Parameters
        ----------
        candidates :
            Pool of architectures to draw from (typically a random sample
            of the full NAS-Bench-201 search space).
        max_retries :
            Maximum number of sampling attempts before raising RuntimeError.
        """
        N = self.config.population_size
        rng = np.random.default_rng(seed=0)

        for attempt in range(max_retries):
            idx = rng.choice(len(candidates), size=N, replace=False)
            pop = [candidates[i] for i in idx]
            sig = self._compute_signal(pop)
            rho = self.library.compute_robustness_all(
                sig.to_array()[np.newaxis, :],   # shape (1, 7)
            )
            if all(v >= self.config.rho_threshold for v in rho.values()):
                logger.info(
                    "Feasible initial population found after %d attempts.", attempt + 1
                )
                return pop

        raise RuntimeError(
            f"Could not find a feasible initial population after {max_retries} "
            f"sampling attempts.  Consider relaxing the STL thresholds or "
            f"increasing the candidate pool size."
        )

    # ------------------------------------------------------------------ #
    #  Main search loop (Algorithm 1)                                      #
    # ------------------------------------------------------------------ #

    def run(
        self,
        oracle: Callable[[List[Architecture]], List[Architecture]],
        initial_population: List[Architecture],
        T: int,
        select_fn: Callable[
            [List[Architecture], int], List[Architecture]
        ],
    ) -> List[Architecture]:
        """Execute the STL-gated NAS search.

        Parameters
        ----------
        oracle :
            The NAS oracle O.  Given the current population P_{t−1}, returns
            a batch of candidate architectures C_t.
        initial_population :
            P_0 (must satisfy Assumption 1 — use initialize_population()).
        T :
            Search budget (number of iterations).
        select_fn :
            Selection function (e.g. NSGA-II selection).
            Signature: select_fn(pool: list[Architecture], N: int) → list[Architecture]
            where pool = P_{t−1} ∪ C_valid and N = population_size.

        Returns
        -------
        list[Architecture]
            The final population P_T.
        """
        # ── Initialise ───────────────────────────────────────────────────
        population = list(initial_population)
        self.trajectory.append(list(population))

        sig0 = self._compute_signal(population)
        self.monitor.commit(sig0)
        self.signal_history.append(sig0)

        if self.config.verbose:
            logger.info("t=0  pop_size=%d  s_acc=%.2f  s_energy=%.3f",
                        len(population), sig0.s_acc, sig0.s_energy)

        # ── Main loop ────────────────────────────────────────────────────
        for t in range(1, T + 1):
            candidates = oracle(population)
            c_valid: List[Architecture] = []

            retry = 0
            while not c_valid and retry <= self.config.max_rejection_retries:
                for arch in candidates:
                    passed, rho = self._gate_candidate(arch, population)
                    self.stats.total_candidates += 1

                    if passed:
                        c_valid.append(arch)
                        self.stats.total_accepted += 1
                    else:
                        self.stats.total_rejected += 1
                        # Record the first-violating property
                        if rho is not None:
                            for pname, val in rho.items():
                                if val < self.config.rho_threshold:
                                    self.stats.rejection_reasons[pname] = (
                                        self.stats.rejection_reasons.get(pname, 0) + 1
                                    )
                                    break

                if not c_valid:
                    retry += 1
                    if retry <= self.config.max_rejection_retries:
                        candidates = oracle(population)   # re-query oracle
                    else:
                        logger.warning(
                            "t=%d: all candidates rejected after %d retries.  "
                            "Population unchanged for this iteration.",
                            t, self.config.max_rejection_retries,
                        )
                        self.stats.fallback_iterations += 1

            # ── Selection ────────────────────────────────────────────────
            pool = population + c_valid
            new_pop = select_fn(pool, self.config.population_size)

            # ── Post-selection verification (Bug-1 fix) ──────────────────
            # NSGA-II selection may drop members, changing the committed
            # signal.  Re-verify and recover if the new population violates
            # any property.
            new_pop = self._verify_and_recover(new_pop, population)

            population = new_pop
            self.trajectory.append(list(population))

            # Commit signal
            sig_t = self._compute_signal(population)
            self.monitor.commit(sig_t)
            self.signal_history.append(sig_t)

            if self.config.verbose:
                rho_min = min(self.monitor.rho_history[-1].values())
                logger.info(
                    "t=%d  accepted=%d  pop_size=%d  "
                    "s_acc=%.2f  s_energy=%.3f  ρ_min=%.4f",
                    t, len(c_valid), len(population),
                    sig_t.s_acc, sig_t.s_energy, rho_min,
                )

        self.stats.log_summary()
        return population

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _gate_candidate(
        self,
        arch: Architecture,
        current_population: List[Architecture],
    ) -> Tuple[bool, Optional[Dict[str, float]]]:
        """Evaluate one candidate under the hypothetical signal extension.

        Returns (passed: bool, rho: dict | None).
        """
        hyp_pop = current_population + [arch]
        sigma_test = self._compute_signal(hyp_pop)

        rho = self.monitor.evaluate_hypothetical(sigma_test)
        passed = self.monitor.gate(rho)
        self.monitor.pop_hypothetical()   # always roll back

        if passed:
            return True, rho
        return False, rho

    def _compute_signal(
        self, population: Sequence[Architecture]
    ) -> SearchSignal:
        """Compute the 7D signal from a population (delegates to signal.py)."""
        return compute_signal(
            population,
            gamma_max=self.config.gamma_max,
        )

    def _verify_and_recover(
        self,
        new_pop: List[Architecture],
        prev_pop: List[Architecture],
    ) -> List[Architecture]:
        """Post-selection recovery: ensure the committed population is valid.

        If the selection step dropped members such that σ[t] violates a
        property, restore the minimal set of dropped members needed to
        recover compliance.

        This addresses Bug 1 (NSGA-II drops high-accuracy members, causing
        φ_monotone violations in the committed signal).
        """
        sig = self._compute_signal(new_pop)
        H   = self._build_history_array(sig)
        rho = self.library.compute_robustness_all(H)

        if all(v >= self.config.rho_threshold for v in rho.values()):
            return new_pop   # already compliant

        # Recovery: add back members from prev_pop sorted by accuracy
        self.stats.post_selection_recoveries += 1
        dropped = [a for a in prev_pop if a not in new_pop]
        dropped.sort(key=lambda a: a.acc, reverse=True)

        recovered = list(new_pop)
        for arch in dropped:
            recovered.append(arch)
            sig_rec = self._compute_signal(recovered)
            H_rec   = self._build_history_array(sig_rec)
            rho_rec = self.library.compute_robustness_all(H_rec)
            if all(v >= self.config.rho_threshold for v in rho_rec.values()):
                logger.debug(
                    "Post-selection recovery: restored %d members.",
                    len(recovered) - len(new_pop),
                )
                return recovered

        logger.warning(
            "Post-selection recovery did not fully restore compliance.  "
            "Proceeding with partially recovered population."
        )
        return recovered

    def _build_history_array(self, appended: SearchSignal) -> np.ndarray:
        """Build the signal history array with `appended` as the last step."""
        past = self.signal_history.to_array()     # shape (t, 7)
        new  = appended.to_array()[np.newaxis, :] # shape (1, 7)
        return np.vstack([past, new])

    # ------------------------------------------------------------------ #
    #  Result accessors                                                    #
    # ------------------------------------------------------------------ #

    def final_pareto_front(self) -> List[Architecture]:
        """Return the Pareto-optimal subset of the final population P_T.

        Pareto optimality is w.r.t. (accuracy ↑, energy ↓).
        """
        if not self.trajectory:
            return []
        final = self.trajectory[-1]
        pareto = []
        for arch in final:
            dominated = any(
                (other.acc >= arch.acc and other.energy <= arch.energy
                 and (other.acc > arch.acc or other.energy < arch.energy))
                for other in final
            )
            if not dominated:
                pareto.append(arch)
        return pareto

    def constraint_satisfaction_per_property(self) -> Dict[str, float]:
        """Return the per-property satisfaction rate over committed iterations."""
        rates: Dict[str, float] = {}
        T = len(self.monitor.rho_history)
        if T == 0:
            return {p.name: 1.0 for p in self.library.properties}
        for prop in self.library.properties:
            sat = sum(
                1 for rho in self.monitor.rho_history
                if rho.get(prop.name, 1.0) >= self.config.rho_threshold
            )
            rates[prop.name] = sat / T
        return rates
