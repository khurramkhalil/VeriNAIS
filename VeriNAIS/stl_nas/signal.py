"""
stl_nas/signal.py
-----------------
Seven-dimensional search signal σ_τ(t) as defined in Section III of the paper.

Each dimension is a real-valued scalar computed from population statistics at
search iteration t.  The signal is used by the STL monitor to evaluate all
eight STL properties.

Signal dimensions (Eq. 2)
--------------------------
  s_acc(t)     mean test accuracy across the population                 (%)
  s_energy(t)  mean FLOP-normalised energy proxy ∈ [0, 2]
  s_params(t)  mean parameter count / γ_max  (dimensionless ratio)
  s_conv(t)    fraction of convolutional edges in the population        [0, 1]
  s_hv(t)      hypervolume of the Pareto front                         (diagnostic)
  s_lat(t)     max Jetson-Thor latency across the population           (ms)
  s_pwr(t)     max Jetson-Thor power draw across the population        (W)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Data container for one time-step of the search signal
# ---------------------------------------------------------------------------

@dataclass
class SearchSignal:
    """One time-step of the 7D search signal σ_τ(t).

    Parameters match the signal definition in Eq. (2) of the paper.
    """
    s_acc: float       # mean test accuracy (%)
    s_energy: float    # mean FLOP-normalised energy ∈ [0, 2]
    s_params: float    # mean params / γ_max  (dimensionless ratio)
    s_conv: float      # conv-op fraction in [0, 1]
    s_hv: float        # Pareto-front hypervolume (diagnostic only)
    s_lat: float       # max latency [ms] from Jetson Thor lookup
    s_pwr: float       # max power [W] from Jetson Thor lookup

    def to_array(self) -> np.ndarray:
        """Return the signal as a (7,) numpy array in canonical order."""
        return np.array([
            self.s_acc,
            self.s_energy,
            self.s_params,
            self.s_conv,
            self.s_hv,
            self.s_lat,
            self.s_pwr,
        ], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "SearchSignal":
        arr = np.asarray(arr, dtype=np.float64)
        if arr.shape != (7,):
            raise ValueError(f"Expected shape (7,), got {arr.shape}")
        return cls(*arr.tolist())

    # Dimension names in canonical order — used by STLMonitor to name RTAMT vars
    DIMS: List[str] = field(default_factory=lambda: [
        "s_acc", "s_energy", "s_params", "s_conv",
        "s_hv", "s_lat", "s_pwr",
    ])


SearchSignal.DIMS = [
    "s_acc", "s_energy", "s_params", "s_conv",
    "s_hv", "s_lat", "s_pwr",
]


# ---------------------------------------------------------------------------
# Architecture representation (thin wrapper around NAS-Bench-201 index)
# ---------------------------------------------------------------------------

@dataclass
class Architecture:
    """A single NAS-Bench-201 architecture.

    Attributes
    ----------
    arch_id : int
        Index in NAS-Bench-201 / NATS-Bench TSS (0 … 15 624).
    acc : float
        Test accuracy on the chosen dataset (%) from the benchmark table.
    energy : float
        FLOP-normalised energy proxy ∈ [0, 2].
        Computed as:  energy = FLOPs(A) / FLOPs_max
        where FLOPs_max = 220.1 M for the conv-heavy reference architecture
        on CIFAR-10 (32×32 input).
    params : int
        Raw parameter count (number of trainable weights).
    conv_fraction : float
        Fraction of the six cell edges that are convolutional operations
        (nor_conv_1x1 or nor_conv_3x3).
    lat_ms : float
        Measured latency on Jetson Thor (TensorRT FP16) in milliseconds.
        Retrieved from the hw_table lookup or bilinear interpolation.
    pwr_w : float
        Measured power on Jetson Thor (tegrastats) in Watts.
        Retrieved from the hw_table lookup or bilinear interpolation.
    """
    arch_id: int
    acc: float
    energy: float
    params: int
    conv_fraction: float
    lat_ms: float
    pwr_w: float


# ---------------------------------------------------------------------------
# Signal computation from a population of architectures
# ---------------------------------------------------------------------------

def pareto_hypervolume(
    population: Sequence[Architecture],
    ref_point: tuple[float, float] = (0.0, 0.0),
) -> float:
    """Compute the 2-objective hypervolume of the Pareto front.

    Objectives are accuracy (maximise) and negative energy (maximise, i.e.
    minimise energy).  The reference point is set at (acc=0, -energy=0).

    This implements the sweep-line algorithm for 2D hypervolume, which runs
    in O(n log n).

    Parameters
    ----------
    population :
        Set of architectures in the current population.
    ref_point :
        (acc_ref, neg_energy_ref) reference point — must be dominated by all
        Pareto-front members for a positive HV.

    Returns
    -------
    float
        Hypervolume indicator value.
    """
    if not population:
        return 0.0

    # Build objective vectors: (acc, -energy) — both to be maximised
    points = [(a.acc, -a.energy) for a in population]

    # Extract Pareto front (non-dominated set)
    pareto: list[tuple[float, float]] = []
    for p in points:
        dominated = False
        for q in points:
            if q[0] >= p[0] and q[1] >= p[1] and q != p:
                dominated = True
                break
        if not dominated:
            pareto.append(p)

    if not pareto:
        return 0.0

    # Sort by first objective descending
    pareto.sort(key=lambda x: x[0], reverse=True)

    # Sweep-line HV in 2D
    hv = 0.0
    prev_y = ref_point[1]
    for x, y in pareto:
        if y > prev_y:
            hv += (x - ref_point[0]) * (y - prev_y)
            prev_y = y

    return hv


def compute_signal(
    population: Sequence[Architecture],
    gamma_max: float = 500_000.0,
    hv_ref_point: tuple[float, float] = (0.0, 0.0),
) -> SearchSignal:
    """Compute the 7D search signal from a population snapshot.

    Called at every NAS iteration t (both for the committed signal σ[t] and
    for the hypothetical extension σ_test used by the gatekeeper).

    Parameters
    ----------
    population :
        All architectures currently in the population P_t.
    gamma_max :
        Parameter count budget (used to normalise s_params).
        Default: 500 000 (34 % of NATSBench architectures exceed this).
    hv_ref_point :
        Reference point for Pareto-HV computation.

    Returns
    -------
    SearchSignal
        The 7D signal vector at this iteration.
    """
    if not population:
        return SearchSignal(
            s_acc=0.0, s_energy=0.0, s_params=0.0,
            s_conv=0.0, s_hv=0.0, s_lat=0.0, s_pwr=0.0,
        )

    n = len(population)

    s_acc    = sum(a.acc for a in population) / n
    s_energy = sum(a.energy for a in population) / n
    s_params = (sum(a.params for a in population) / n) / gamma_max
    s_conv   = sum(a.conv_fraction for a in population) / n
    s_hv     = pareto_hypervolume(population, ref_point=hv_ref_point)
    s_lat    = max(a.lat_ms for a in population)
    s_pwr    = max(a.pwr_w for a in population)

    return SearchSignal(
        s_acc=s_acc,
        s_energy=s_energy,
        s_params=s_params,
        s_conv=s_conv,
        s_hv=s_hv,
        s_lat=s_lat,
        s_pwr=s_pwr,
    )


# ---------------------------------------------------------------------------
# Signal trajectory (list of SearchSignal, one per iteration)
# ---------------------------------------------------------------------------

class SignalHistory:
    """Maintains the growing signal prefix σ[0 : t] for online monitoring.

    Used by the gatekeeper to append the hypothetical extension and then roll
    it back if the candidate is rejected.
    """

    def __init__(self) -> None:
        self._history: list[SearchSignal] = []

    def __len__(self) -> int:
        return len(self._history)

    def __getitem__(self, idx):
        return self._history[idx]

    def append(self, sig: SearchSignal) -> None:
        self._history.append(sig)

    def pop(self) -> SearchSignal:
        return self._history.pop()

    def to_array(self) -> np.ndarray:
        """Return the full history as shape (T, 7) array."""
        if not self._history:
            return np.empty((0, 7), dtype=np.float64)
        return np.stack([s.to_array() for s in self._history], axis=0)

    def get_dim(self, dim_name: str) -> np.ndarray:
        """Return the time-series for a single signal dimension as a 1-D array."""
        idx = SearchSignal.DIMS.index(dim_name)
        return self.to_array()[:, idx]
