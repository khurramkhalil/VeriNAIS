"""
stl_nas/algorithms/random_search.py
------------------------------------
Random search oracle and selection for the STL-NAS gatekeeper.

Random search samples architectures uniformly at random from the NAS-Bench-201
search space at each iteration, without any learning or population-guided
proposal.  It serves as the baseline against which all other NAS paradigms
are compared in Tables 1–3 of the paper.

In the STL-NAS gated condition, random search becomes a rejection sampler:
architectures are drawn uniformly from the full search space and accepted
only if they satisfy all STL properties under the hypothetical signal
extension.  This provides the lower bound on what the gatekeeper's STL
guarantees cost in terms of search efficiency.

Selection
---------
For random search, selection simply retains the N architectures with the
highest hypervolume contribution on the Pareto front.  This is equivalent
to NSGA-II selection without the evolutionary operator; it is included
separately to isolate the effect of the oracle from the selection mechanism.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..signal import Architecture


# ---------------------------------------------------------------------------
# Random oracle
# ---------------------------------------------------------------------------

class RandomOracle:
    """Uniform random architecture sampler.

    At each call, draws `candidates_per_iter` architectures independently
    and uniformly at random from the full NAS-Bench-201 search space.

    Parameters
    ----------
    arch_pool :
        All 15 625 architectures in the NAS-Bench-201 search space (pre-loaded).
    candidates_per_iter :
        Batch size (number of candidates proposed per iteration).
    seed :
        Random seed for reproducibility.
    """

    def __init__(
        self,
        arch_pool: List[Architecture],
        candidates_per_iter: int = 20,
        seed: Optional[int] = None,
    ) -> None:
        self.arch_pool           = arch_pool
        self.candidates_per_iter = candidates_per_iter
        self._rng                = np.random.default_rng(seed)

    def __call__(self, population: List[Architecture]) -> List[Architecture]:
        """Return a uniformly sampled batch (population argument is ignored)."""
        idx = self._rng.choice(
            len(self.arch_pool),
            size=self.candidates_per_iter,
            replace=False,
        )
        return [self.arch_pool[i] for i in idx]


# ---------------------------------------------------------------------------
# Selection: Hypervolume-contribution based
# ---------------------------------------------------------------------------

def _hypervolume_contribution(
    arch: Architecture,
    population: List[Architecture],
    ref_point: tuple[float, float] = (0.0, 0.0),
) -> float:
    """Compute the hypervolume contribution of `arch` relative to `population`.

    HV contribution = HV(population ∪ {arch}) − HV(population).
    Computed in the (acc, −energy) objective space (both maximised).
    """
    def hv_2d(points: List[tuple[float, float]]) -> float:
        if not points:
            return 0.0
        # Extract non-dominated points, sort by first objective desc
        pareto: List[tuple[float, float]] = []
        for p in points:
            if not any(
                q[0] >= p[0] and q[1] >= p[1] and q != p for q in points
            ):
                pareto.append(p)
        if not pareto:
            return 0.0
        pareto.sort(key=lambda x: x[0], reverse=True)
        hv, prev_y = 0.0, ref_point[1]
        for x, y in pareto:
            if y > prev_y:
                hv += (x - ref_point[0]) * (y - prev_y)
                prev_y = y
        return hv

    pts_without = [(a.acc, -a.energy) for a in population]
    pts_with    = pts_without + [(arch.acc, -arch.energy)]

    return hv_2d(pts_with) - hv_2d(pts_without)


def random_selection(
    pool: List[Architecture],
    N: int,
) -> List[Architecture]:
    """Select N architectures from pool by hypervolume contribution.

    Greedily adds architectures in decreasing order of marginal HV
    contribution (a standard greedy Pareto selection rule).

    This is used by the random search oracle as its selection function.
    Other NAS algorithms use nsga2_selection (see algorithms/nsga2.py).

    Parameters
    ----------
    pool :
        Combined set P_{t-1} ∪ C_valid.
    N :
        Target population size.

    Returns
    -------
    list[Architecture]
    """
    if len(pool) <= N:
        return list(pool)

    selected: List[Architecture] = []
    remaining = list(pool)

    while len(selected) < N and remaining:
        best_idx  = max(
            range(len(remaining)),
            key=lambda i: _hypervolume_contribution(remaining[i], selected),
        )
        selected.append(remaining.pop(best_idx))

    return selected
