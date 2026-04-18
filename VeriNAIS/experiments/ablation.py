"""
experiments/ablation.py
-----------------------
Property-subset ablation study (Section VII-F, Table 5).

Reproduces the ablation in which individual STL properties are removed one
at a time to measure their contribution to constraint satisfaction and
Pareto-front hypervolume.  Also performs the β_max sensitivity sweep.

Ablation conditions
-------------------
  full_phi          : all 8 properties enabled (default STL-NAS)
  no_energy         : φ_energy disabled
  no_size           : φ_size disabled
  no_latency        : φ_latency disabled
  no_power          : φ_power disabled
  no_monotone       : φ_monotone disabled
  no_diversity      : φ_diversity disabled
  no_recovery       : φ_recovery disabled     (temporal)
  no_stagnation     : φ_stagnation disabled   (temporal)
  no_temporal       : both temporal properties disabled
  resource_only     : only the 4 resource-bound properties
  dynamics_only     : only the 2 search-dynamics properties

β_max sensitivity sweep (RQ4, Figure in paper)
------------------------------------------------
  β_max ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4}
  For each value, run NSGA-II gated on CIFAR-100, 10 trials.
  Report: final HV, overall_sat_rate, phi_energy sat_rate.

Usage
-----
  python experiments/ablation.py \
      --dataset cifar100 \
      --paradigm nsga2 \
      --trials 10 \
      --seed 0 \
      --output results/ablation/
"""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np

from stl_nas import STLPropertyLibrary, DEFAULT_THRESHOLDS
from experiments.run_nas import run_trial
from data.nasbench201 import NATSBenchLoader
from hardware.hw_table import HardwareLookupTable


# ---------------------------------------------------------------------------
# Ablation subset definitions
# ---------------------------------------------------------------------------

ALL_PROPS = [
    "phi_energy", "phi_size", "phi_latency", "phi_power",
    "phi_monotone", "phi_diversity", "phi_recovery", "phi_stagnation",
]

ABLATION_SETS: Dict[str, List[str]] = {
    "full_phi":       ALL_PROPS,
    "no_energy":      [p for p in ALL_PROPS if p != "phi_energy"],
    "no_size":        [p for p in ALL_PROPS if p != "phi_size"],
    "no_latency":     [p for p in ALL_PROPS if p != "phi_latency"],
    "no_power":       [p for p in ALL_PROPS if p != "phi_power"],
    "no_monotone":    [p for p in ALL_PROPS if p != "phi_monotone"],
    "no_diversity":   [p for p in ALL_PROPS if p != "phi_diversity"],
    "no_recovery":    [p for p in ALL_PROPS if p != "phi_recovery"],
    "no_stagnation":  [p for p in ALL_PROPS if p != "phi_stagnation"],
    "no_temporal":    [p for p in ALL_PROPS
                       if p not in ("phi_recovery", "phi_stagnation")],
    "resource_only":  ["phi_energy", "phi_size", "phi_latency", "phi_power"],
    "dynamics_only":  ["phi_monotone", "phi_diversity"],
}

BETA_MAX_SWEEP = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]


# ---------------------------------------------------------------------------
# STLPropertyLibrary subset factory
# ---------------------------------------------------------------------------

class SubsetSTLPropertyLibrary(STLPropertyLibrary):
    """STLPropertyLibrary that exposes only a named subset of properties.

    Used by the ablation to disable individual properties without modifying
    the monitor or gatekeeper logic.
    """

    def __init__(
        self,
        active_props: List[str],
        thresholds: Optional[Dict] = None,
    ) -> None:
        super().__init__(thresholds=thresholds)
        # Filter to only active properties
        self.properties = [p for p in self.properties if p.name in active_props]


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------

def run_ablation(
    arch_pool,
    paradigm: str,
    dataset: str,
    trials: int,
    T: int,
    pop_size: int,
    seed: int,
    rho_th: float,
    hw_lookup: HardwareLookupTable,
    output_dir: str,
) -> None:
    """Run all property-subset ablation conditions."""
    os.makedirs(output_dir, exist_ok=True)

    for condition_name, active_props in ABLATION_SETS.items():
        print(f"\n[ABLATION] condition={condition_name}  "
              f"active_props={active_props}")

        library  = SubsetSTLPropertyLibrary(active_props)
        results  = []

        for trial in range(trials):
            s = seed + trial
            r = run_trial(
                arch_pool=arch_pool,
                paradigm=paradigm,
                gated=True,
                T=T,
                pop_size=pop_size,
                seed=s,
                rho_th=rho_th,
                library=library,
                hw_lookup=hw_lookup,
            )
            results.append(r)
            print(f"  trial {trial + 1}/{trials}  HV={r['final_hv']:.4f}  "
                  f"sat={r['overall_sat_rate']:.2%}")

        out_file = os.path.join(
            output_dir,
            f"ablation_{paradigm}_{dataset}_{condition_name}.json",
        )
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved → {out_file}")


def run_beta_sweep(
    arch_pool,
    paradigm: str,
    dataset: str,
    trials: int,
    T: int,
    pop_size: int,
    seed: int,
    rho_th: float,
    hw_lookup: HardwareLookupTable,
    output_dir: str,
) -> None:
    """β_max sensitivity sweep."""
    sweep_results = []

    for beta in BETA_MAX_SWEEP:
        th = dict(DEFAULT_THRESHOLDS)
        th["beta_max"] = beta
        library = STLPropertyLibrary(thresholds=th)
        trial_results = []

        for trial in range(trials):
            s = seed + trial
            r = run_trial(
                arch_pool=arch_pool,
                paradigm=paradigm,
                gated=True,
                T=T,
                pop_size=pop_size,
                seed=s,
                rho_th=rho_th,
                library=library,
                hw_lookup=hw_lookup,
            )
            trial_results.append(r)

        mean_hv  = float(np.mean([r["final_hv"] for r in trial_results]))
        mean_sat = float(np.mean([r["overall_sat_rate"] for r in trial_results]))
        e_sat    = float(np.mean([
            r["sat_rates"].get("phi_energy", 1.0) for r in trial_results
        ]))

        sweep_results.append({
            "beta_max":     beta,
            "mean_hv":      round(mean_hv,  4),
            "mean_sat":     round(mean_sat,  4),
            "energy_sat":   round(e_sat,     4),
        })
        print(f"  β_max={beta:.1f}  HV={mean_hv:.4f}  "
              f"sat={mean_sat:.2%}  E_sat={e_sat:.2%}")

    out_file = os.path.join(
        output_dir, f"beta_sweep_{paradigm}_{dataset}.json"
    )
    with open(out_file, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"[β sweep] Saved → {out_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="STL-NAS property ablation study.")
    p.add_argument("--paradigm",      default="nsga2")
    p.add_argument("--dataset",       default="cifar100")
    p.add_argument("--trials",        type=int,   default=10)
    p.add_argument("--T",             type=int,   default=50)
    p.add_argument("--pop_size",      type=int,   default=20)
    p.add_argument("--seed",          type=int,   default=0)
    p.add_argument("--rho_th",        type=float, default=0.0)
    p.add_argument("--nasbench_path", default="data/nasbench201/NATS-tss-v1_0-3ffb9.pickle.pbz2")
    p.add_argument("--output",        default="results/ablation/")
    p.add_argument("--beta_sweep",    action="store_true",
                   help="Also run the β_max sensitivity sweep")
    args = p.parse_args()

    hw_lookup = HardwareLookupTable()
    loader    = NATSBenchLoader(args.nasbench_path, dataset=args.dataset,
                                hw_lookup=hw_lookup)
    print("[INFO] Loading NAS-Bench-201 …")
    arch_pool = loader.load()
    print(f"[INFO] Loaded {len(arch_pool)} architectures.")

    run_ablation(
        arch_pool=arch_pool,
        paradigm=args.paradigm,
        dataset=args.dataset,
        trials=args.trials,
        T=args.T,
        pop_size=args.pop_size,
        seed=args.seed,
        rho_th=args.rho_th,
        hw_lookup=hw_lookup,
        output_dir=args.output,
    )

    if args.beta_sweep:
        run_beta_sweep(
            arch_pool=arch_pool,
            paradigm=args.paradigm,
            dataset=args.dataset,
            trials=args.trials,
            T=args.T,
            pop_size=args.pop_size,
            seed=args.seed,
            rho_th=args.rho_th,
            hw_lookup=hw_lookup,
            output_dir=args.output,
        )


if __name__ == "__main__":
    main()
