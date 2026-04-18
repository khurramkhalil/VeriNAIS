"""
stl_nas/algorithms/
-------------------
NAS oracle wrappers for the four paradigms evaluated in the paper:

  nsga2.py         -- NSGA-II (evolutionary multi-objective)
  enas.py          -- ENAS   (reinforcement learning, controller-based)
  lamoo.py         -- LaMOO  (Bayesian optimisation with latent space)
  random_search.py -- Random search baseline

Each module exposes:
  - An Oracle class whose __call__(population) method returns a list of
    candidate Architecture objects for the gatekeeper to screen.
  - A select() function used by the gatekeeper's select_fn parameter
    (performs algorithm-specific selection after gating).

All oracles operate over the NAS-Bench-201 search space
(|A| = 15 625 architectures) and retrieve accuracy/energy from the
pre-loaded benchmark table via the NATSBenchAPI helper.
"""

from .nsga2 import NSGA2Oracle, nsga2_selection
from .enas import ENASOracle
from .lamoo import LaMOOOracle
from .random_search import RandomOracle, random_selection

__all__ = [
    "NSGA2Oracle",
    "nsga2_selection",
    "ENASOracle",
    "LaMOOOracle",
    "RandomOracle",
    "random_selection",
]
