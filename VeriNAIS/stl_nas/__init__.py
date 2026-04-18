"""
STL-NAS: Formally Verified Neural Architecture Search
for Safety-Critical Cyber-Physical Systems.

EMSOFT 2026 / IEEE TCAD Supplementary Material

Package layout
--------------
stl_nas/
  signal.py        -- 7-dimensional search signal computation
  properties.py    -- Eight STL properties (resource bounds, search
                      dynamics, temporal interaction)
  monitor.py       -- RTAMT-backed online STL monitor
  gatekeeper.py    -- STL gatekeeper (Algorithm 1 from the paper)
  algorithms/      -- NAS oracle wrappers (NSGA-II, ENAS, LaMOO, Random)

data/
  nasbench201.py   -- NAS-Bench-201 / NATS-Bench TSS loader
  cifar10.py       -- CIFAR-10 dataset loader
  cifar100.py      -- CIFAR-100 dataset loader
  imagenet16.py    -- ImageNet-16-120 dataset loader

hardware/
  hw_table.py      -- Jetson Thor latency/power lookup table
  measure_harness.py -- On-device measurement harness (tegrastats)

experiments/
  run_nas.py       -- Main experiment entry point
  ablation.py      -- Property-subset ablation (Section VII-F)
  hw_deploy.py     -- Hardware deployment experiments (Table 4)
"""

from .signal import SearchSignal, compute_signal
from .properties import STLPropertyLibrary, DEFAULT_THRESHOLDS
from .monitor import STLMonitor
from .gatekeeper import STLGatekeeper

__all__ = [
    "SearchSignal",
    "compute_signal",
    "STLPropertyLibrary",
    "DEFAULT_THRESHOLDS",
    "STLMonitor",
    "STLGatekeeper",
]
