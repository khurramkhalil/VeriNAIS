"""
experiments/run_nas.py
----------------------
Main experiment entry point for STL-NAS.

Reproduces the results in Tables 1–3 of the paper (RQ1–RQ3):
  - Four NAS paradigms: NSGA-II, ENAS, LaMOO, Random
  - Two conditions per paradigm: gated (STL-NAS) and ungated
  - Three datasets: CIFAR-10, CIFAR-100, ImageNet-16-120
  - 10 independent trials per (paradigm, condition, dataset) combination
  - 50 search iterations per trial; population size 20

Usage
-----
  python experiments/run_nas.py \
      --paradigm nsga2 \
      --dataset cifar10 \
      --gated \
      --trials 10 \
      --T 50 \
      --seed 0 \
      --output results/

Output
------
  results/{paradigm}_{dataset}_{gated|ungated}_seed{seed}.json
  Each file contains:
    - per-iteration constraint satisfaction for all 8 properties
    - Pareto-front hypervolume at each iteration
    - final Pareto-front architectures (arch_id, acc, energy)
    - gatekeeper acceptance rate and violation breakdown
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import List, Optional

import numpy as np

from stl_nas import (
    STLPropertyLibrary,
    STLGatekeeper,
    DEFAULT_THRESHOLDS,
)
from stl_nas.gatekeeper import GatekeeperConfig
from stl_nas.algorithms.nsga2 import NSGA2Oracle, nsga2_selection
from stl_nas.algorithms.enas import ENASOracle
from stl_nas.algorithms.lamoo import LaMOOOracle
from stl_nas.algorithms.random_search import RandomOracle, random_selection
from stl_nas.signal import Architecture, compute_signal

from data.nasbench201 import NATSBenchLoader
from hardware.hw_table import HardwareLookupTable


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

PARADIGMS = ("nsga2", "enas", "lamoo", "random")
DATASETS  = ("cifar10", "cifar100", "ImageNet16-120")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run STL-NAS or ungated NAS on NAS-Bench-201."
    )
    p.add_argument("--paradigm",  choices=PARADIGMS, default="nsga2")
    p.add_argument("--dataset",   choices=DATASETS,  default="cifar10")
    p.add_argument("--gated",     action="store_true",
                   help="Run STL-NAS gated condition (default: ungated)")
    p.add_argument("--trials",    type=int,   default=10)
    p.add_argument("--T",         type=int,   default=50,
                   help="Number of search iterations per trial")
    p.add_argument("--pop_size",  type=int,   default=20)
    p.add_argument("--seed",      type=int,   default=0)
    p.add_argument("--rho_th",    type=float, default=0.0,
                   help="STL robustness threshold ρ_th")
    p.add_argument("--nasbench_path", type=str,
                   default="data/nasbench201/NATS-tss-v1_0-3ffb9.pickle.pbz2")
    p.add_argument("--output",    type=str, default="results/")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Oracle factory
# ---------------------------------------------------------------------------

def make_oracle(
    paradigm: str,
    arch_pool: List[Architecture],
    candidates_per_iter: int,
    seed: int,
):
    """Instantiate the NAS oracle for the given paradigm."""
    if paradigm == "nsga2":
        return NSGA2Oracle(
            arch_pool=arch_pool,
            candidates_per_iter=candidates_per_iter,
            seed=seed,
        )
    elif paradigm == "enas":
        return ENASOracle(
            arch_pool=arch_pool,
            candidates_per_iter=candidates_per_iter,
            seed=seed,
        )
    elif paradigm == "lamoo":
        return LaMOOOracle(
            arch_pool=arch_pool,
            candidates_per_iter=candidates_per_iter,
            seed=seed,
        )
    elif paradigm == "random":
        return RandomOracle(
            arch_pool=arch_pool,
            candidates_per_iter=candidates_per_iter,
            seed=seed,
        )
    raise ValueError(f"Unknown paradigm: {paradigm}")


def make_select_fn(paradigm: str):
    """Return the selection function for the given paradigm."""
    if paradigm == "random":
        return random_selection
    return nsga2_selection   # NSGA-II selection for all population-based methods


# ---------------------------------------------------------------------------
# Single-trial execution
# ---------------------------------------------------------------------------

def run_trial(
    arch_pool: List[Architecture],
    paradigm: str,
    gated: bool,
    T: int,
    pop_size: int,
    seed: int,
    rho_th: float,
    library: STLPropertyLibrary,
    hw_lookup: HardwareLookupTable,
) -> dict:
    """Execute one trial and return results as a serialisable dict."""
    rng = np.random.default_rng(seed)

    oracle    = make_oracle(paradigm, arch_pool, pop_size, seed)
    select_fn = make_select_fn(paradigm)

    if gated:
        # ── STL-NAS gated condition ──────────────────────────────────────
        config = GatekeeperConfig(
            rho_threshold=rho_th,
            gamma_max=DEFAULT_THRESHOLDS["gamma_max_ratio"] * 500_000,
            population_size=pop_size,
        )
        gatekeeper = STLGatekeeper(
            library=library,
            config=config,
            hw_lookup=lambda aid: hw_lookup.query(aid),
        )

        # Feasibility-checked initialisation (Assumption 1)
        init_pool = rng.choice(arch_pool, size=5 * pop_size, replace=False).tolist()
        initial_pop = gatekeeper.initialize_population(init_pool)

        t_start = time.perf_counter()
        final_pop = gatekeeper.run(
            oracle=oracle,
            initial_population=initial_pop,
            T=T,
            select_fn=select_fn,
        )
        elapsed = time.perf_counter() - t_start

        # Collect results
        rho_hist   = gatekeeper.monitor.rho_history
        sat_rates  = gatekeeper.constraint_satisfaction_per_property()
        acc_rate   = gatekeeper.stats.acceptance_rate
        violations = gatekeeper.stats.rejection_reasons

    else:
        # ── Ungated baseline ─────────────────────────────────────────────
        init_idx   = rng.choice(len(arch_pool), size=pop_size, replace=False)
        population = [arch_pool[i] for i in init_idx]

        rho_hist   = []
        elapsed    = 0.0

        t_start = time.perf_counter()
        for t in range(T):
            candidates = oracle(population)
            pool       = population + list(candidates)
            population = select_fn(pool, pop_size)

            # Compute signal for logging (no gating)
            sig = compute_signal(population)
            H   = np.array([[
                sig.s_acc, sig.s_energy, sig.s_params, sig.s_conv,
                sig.s_hv, sig.s_lat, sig.s_pwr,
            ]])
            rho = library.compute_robustness_all(H)
            rho_hist.append(rho)

        elapsed    = time.perf_counter() - t_start
        final_pop  = population
        sat_rates  = {}
        for prop in library.properties:
            sat = sum(1 for r in rho_hist if r.get(prop.name, 1.0) >= rho_th)
            sat_rates[prop.name] = sat / len(rho_hist) if rho_hist else 1.0
        acc_rate   = 1.0   # no gating
        violations = {}

    # ── Pareto hypervolume at final iteration ─────────────────────────────
    from stl_nas.signal import pareto_hypervolume
    final_hv = pareto_hypervolume(final_pop)

    return {
        "paradigm":   paradigm,
        "gated":      gated,
        "seed":       seed,
        "T":          T,
        "elapsed_s":  round(elapsed, 3),
        "final_hv":   round(final_hv, 4),
        "sat_rates":  {k: round(v, 4) for k, v in sat_rates.items()},
        "overall_sat_rate": round(
            sum(sat_rates.values()) / len(sat_rates) if sat_rates else 1.0, 4
        ),
        "acceptance_rate": round(acc_rate, 4),
        "rejection_reasons": violations,
        "final_pareto": [
            {"arch_id": a.arch_id, "acc": round(a.acc, 3),
             "energy": round(a.energy, 4)}
            for a in final_pop
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # ── Load benchmark ────────────────────────────────────────────────────
    hw_lookup = HardwareLookupTable()
    loader    = NATSBenchLoader(
        path=args.nasbench_path,
        dataset=args.dataset,
        hw_lookup=hw_lookup,
    )
    print(f"[INFO] Loading NAS-Bench-201 ({args.dataset}) …  (~40 s, ~25 GB RAM)")
    arch_pool = loader.load()
    print(f"[INFO] Loaded {len(arch_pool)} architectures.")

    # ── Build property library ────────────────────────────────────────────
    library = STLPropertyLibrary()

    # ── Run trials ────────────────────────────────────────────────────────
    all_results = []
    condition   = "gated" if args.gated else "ungated"

    for trial in range(args.trials):
        seed = args.seed + trial
        print(f"[INFO] Trial {trial + 1}/{args.trials}  "
              f"paradigm={args.paradigm}  dataset={args.dataset}  "
              f"condition={condition}  seed={seed}")

        result = run_trial(
            arch_pool=arch_pool,
            paradigm=args.paradigm,
            gated=args.gated,
            T=args.T,
            pop_size=args.pop_size,
            seed=seed,
            rho_th=args.rho_th,
            library=library,
            hw_lookup=hw_lookup,
        )
        all_results.append(result)
        print(f"       HV={result['final_hv']:.4f}  "
              f"overall_sat={result['overall_sat_rate']:.2%}  "
              f"elapsed={result['elapsed_s']:.1f}s")

    # ── Save ──────────────────────────────────────────────────────────────
    out_file = os.path.join(
        args.output,
        f"{args.paradigm}_{args.dataset}_{condition}_seed{args.seed}.json",
    )
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[INFO] Results saved to {out_file}")


if __name__ == "__main__":
    main()
