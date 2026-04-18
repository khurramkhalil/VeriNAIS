"""
stl_nas/algorithms/nsga2.py
---------------------------
NSGA-II oracle and selection for the STL-NAS gatekeeper.

NSGA-II (Non-dominated Sorting Genetic Algorithm II, Deb et al. 2002)
is a population-based multi-objective evolutionary algorithm.  In
STL-NAS, it serves as the NAS oracle O: given the current population
P_{t-1}, it proposes a new generation of candidate architectures via
tournament selection, crossover, and mutation over the NAS-Bench-201
operation encoding.

Multi-objective formulation
----------------------------
Objectives (both to be minimised after negation):
  f1 = −acc(A)       maximise accuracy
  f2 =  energy(A)    minimise FLOP-normalised energy proxy

Selection pressure is provided by:
  (i)  Non-domination rank (rank 1 = Pareto front)
  (ii) Crowding distance (tie-break: prefer less crowded regions)

Crossover / Mutation
--------------------
NAS-Bench-201 architectures are represented as a length-6 vector of
operation indices (ops ∈ {0,1,2,3,4}).  We apply:
  - Uniform crossover: each edge independently drawn from either parent
  - Mutation: each edge independently replaced with a random operation
    with probability p_mut = 1/6 (one expected mutation per arch)
"""

from __future__ import annotations

import random
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..signal import Architecture


# ---------------------------------------------------------------------------
# Operation encoding (NAS-Bench-201)
# ---------------------------------------------------------------------------

OPS = [
    "none",
    "skip_connect",
    "nor_conv_1x1",
    "nor_conv_3x3",
    "avg_pool_3x3",
]
N_OPS   = len(OPS)    # 5
N_EDGES = 6           # six edges in the NAS-Bench-201 cell


# ---------------------------------------------------------------------------
# NSGA-II selection
# ---------------------------------------------------------------------------

def _dominates(a: Architecture, b: Architecture) -> bool:
    """Return True if a Pareto-dominates b (acc ↑, energy ↓)."""
    return (a.acc >= b.acc and a.energy <= b.energy
            and (a.acc > b.acc or a.energy < b.energy))


def _non_dominated_sort(
    population: List[Architecture],
) -> List[List[Architecture]]:
    """Partition population into Pareto fronts F_1, F_2, …"""
    n = len(population)
    dominated_by: List[List[int]] = [[] for _ in range(n)]
    domination_count: List[int]   = [0] * n
    fronts: List[List[int]] = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _dominates(population[i], population[j]):
                dominated_by[i].append(j)
            elif _dominates(population[j], population[i]):
                domination_count[i] += 1
        if domination_count[i] == 0:
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front: List[int] = []
        for i in fronts[k]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        fronts.append(next_front)
        k += 1

    return [[population[i] for i in front] for front in fronts[:-1]]


def _crowding_distance(front: List[Architecture]) -> List[float]:
    """Compute crowding distance for each member of a single Pareto front."""
    m = len(front)
    if m <= 2:
        return [np.inf] * m

    dist = [0.0] * m

    for obj_fn in [lambda a: a.acc, lambda a: -a.energy]:
        sorted_idx = sorted(range(m), key=lambda i: obj_fn(front[i]))
        obj_vals   = [obj_fn(front[i]) for i in sorted_idx]
        obj_range  = obj_vals[-1] - obj_vals[0]

        if obj_range == 0:
            continue

        dist[sorted_idx[0]]  = np.inf
        dist[sorted_idx[-1]] = np.inf

        for k in range(1, m - 1):
            dist[sorted_idx[k]] += (
                (obj_vals[k + 1] - obj_vals[k - 1]) / obj_range
            )

    return dist


def nsga2_selection(
    pool: List[Architecture],
    N: int,
) -> List[Architecture]:
    """Select the best N individuals from pool using NSGA-II dominance ranking.

    Parameters
    ----------
    pool :
        Combined set P_{t-1} ∪ C_valid (all architectures eligible for
        selection into the next generation P_t).
    N :
        Target population size.

    Returns
    -------
    list[Architecture]
        The next generation P_t of size min(N, len(pool)).
    """
    fronts   = _non_dominated_sort(pool)
    selected: List[Architecture] = []

    for front in fronts:
        if len(selected) + len(front) <= N:
            selected.extend(front)
        else:
            # Fill remaining slots using crowding distance (prefer less crowded)
            needed  = N - len(selected)
            dists   = _crowding_distance(front)
            ranked  = sorted(range(len(front)), key=lambda i: -dists[i])
            selected.extend(front[ranked[i]] for i in range(needed))
            break

    return selected[:N]


# ---------------------------------------------------------------------------
# NSGA-II oracle
# ---------------------------------------------------------------------------

class NSGA2Oracle:
    """NSGA-II candidate generator.

    At each call, the oracle applies binary tournament selection to the
    current population to choose parents, then produces offspring via
    uniform crossover and mutation.

    Parameters
    ----------
    arch_pool :
        Full NAS-Bench-201 search space (list of pre-loaded Architecture
        objects).  Mutation draws new operation types from this pool.
    candidates_per_iter :
        Number of offspring to generate per call (|C_t|).
    p_mutation :
        Per-edge mutation probability (default 1/6 ≈ one mutation/arch).
    seed :
        Random seed for reproducibility.
    """

    def __init__(
        self,
        arch_pool: List[Architecture],
        candidates_per_iter: int = 20,
        p_mutation: float = 1.0 / N_EDGES,
        seed: Optional[int] = None,
    ) -> None:
        self.arch_pool           = arch_pool
        self.candidates_per_iter = candidates_per_iter
        self.p_mutation          = p_mutation
        self._rng                = np.random.default_rng(seed)

        # Build an index: arch_id → Architecture for O(1) lookup
        self._id_to_arch = {a.arch_id: a for a in arch_pool}

    # ------------------------------------------------------------------
    # Oracle interface
    # ------------------------------------------------------------------

    def __call__(self, population: List[Architecture]) -> List[Architecture]:
        """Generate a new batch of candidate architectures.

        Parameters
        ----------
        population :
            Current population P_{t-1}.

        Returns
        -------
        list[Architecture]
            Candidate batch C_t (before gating).
        """
        candidates: List[Architecture] = []
        for _ in range(self.candidates_per_iter):
            parent_a = self._tournament_select(population)
            parent_b = self._tournament_select(population)
            child_id = self._crossover_mutate(parent_a.arch_id, parent_b.arch_id)
            if child_id in self._id_to_arch:
                candidates.append(self._id_to_arch[child_id])
        return candidates

    # ------------------------------------------------------------------
    # Genetic operators
    # ------------------------------------------------------------------

    def _tournament_select(
        self, population: List[Architecture], k: int = 2
    ) -> Architecture:
        """Binary tournament selection (prefer lower energy, then higher acc)."""
        contestants = self._rng.choice(len(population), size=k, replace=False)
        best = population[contestants[0]]
        for idx in contestants[1:]:
            challenger = population[idx]
            if _dominates(challenger, best):
                best = challenger
            elif not _dominates(best, challenger):
                # Tie: prefer higher accuracy
                if challenger.acc > best.acc:
                    best = challenger
        return best

    def _arch_id_to_ops(self, arch_id: int) -> List[int]:
        """Decode NAS-Bench-201 arch_id (base-5 encoding) to op indices."""
        ops = []
        n = arch_id
        for _ in range(N_EDGES):
            ops.append(n % N_OPS)
            n //= N_OPS
        return ops

    def _ops_to_arch_id(self, ops: Sequence[int]) -> int:
        """Encode op indices back to arch_id (base-5)."""
        arch_id = 0
        for i, op in enumerate(ops):
            arch_id += op * (N_OPS ** i)
        return arch_id

    def _crossover_mutate(self, id_a: int, id_b: int) -> int:
        """Uniform crossover + per-edge mutation → child arch_id."""
        ops_a = self._arch_id_to_ops(id_a)
        ops_b = self._arch_id_to_ops(id_b)

        # Uniform crossover
        child_ops = [
            ops_a[i] if self._rng.random() < 0.5 else ops_b[i]
            for i in range(N_EDGES)
        ]

        # Per-edge mutation
        child_ops = [
            int(self._rng.integers(0, N_OPS)) if self._rng.random() < self.p_mutation
            else op
            for op in child_ops
        ]

        return self._ops_to_arch_id(child_ops)
