"""
experiments/hw_deploy.py
------------------------
Hardware deployment validation experiments (Section VII-E, Table 4).

Deploys the final Pareto-front architectures from gated and ungated NAS
runs onto the NVIDIA Jetson Thor embedded platform and measures:
  - Inference latency (ms), mean ± std over 1 000 runs
  - Power draw (W), mean over inference window (tegrastats @ 100 ms)
  - Energy per inference (mJ) = latency × power / 1 000
  - Peak activation memory (MB)

Experimental protocol (Phase 2B)
----------------------------------
  1. For each (paradigm, condition) pair, load the saved results from
     run_nas.py and extract the top-3 Pareto-front architectures by
     hypervolume contribution.
  2. Build the NAS-Bench-201 cell for each architecture.
  3. Compile to TensorRT FP16 via compile_trt().
  4. Run the measurement pipeline (measure_architecture()) on Jetson Thor.
  5. Aggregate results into Table 4.

Key finding (Section VII-E)
-----------------------------
STL-gated Pareto fronts exhibit tighter variance in latency and power
than ungated fronts, confirming that search-time formal guarantees
translate to predictable deployment behaviour.  Ungated Pareto fronts
contain architectures that violate the real-time latency budget
(λ_max = 5.0 ms) even when they appeared efficient under FLOP proxies.

Usage
-----
  # On Jetson Thor (after copying results from the search machine):
  python experiments/hw_deploy.py \
      --results_dir results/ \
      --paradigms nsga2 enas lamoo random \
      --dataset cifar10 \
      --top_k 3 \
      --output results/hw_deploy/
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from hardware.measure_harness import measure_architecture
from hardware.hw_table import HardwareLookupTable
from data.nasbench201 import NATSBenchLoader, NATSBenchLoader


# ---------------------------------------------------------------------------
# Pareto-front utilities
# ---------------------------------------------------------------------------

def top_k_by_hv_contribution(
    pareto: List[Dict],
    k: int = 3,
) -> List[Dict]:
    """Select the top-k architectures by hypervolume contribution.

    Parameters
    ----------
    pareto :
        List of dicts with keys 'arch_id', 'acc', 'energy'
        (from the 'final_pareto' field of run_nas.py output).
    k :
        Number of architectures to return.

    Returns
    -------
    list[dict]  (up to k entries, sorted by HV contribution desc)
    """
    if len(pareto) <= k:
        return pareto

    from stl_nas.signal import Architecture
    from stl_nas.algorithms.random_search import _hypervolume_contribution

    arch_objs = [
        Architecture(
            arch_id=d["arch_id"],
            acc=d["acc"],
            energy=d["energy"],
            params=0,
            conv_fraction=0.0,
            lat_ms=0.0,
            pwr_w=0.0,
        )
        for d in pareto
    ]

    contributions = [
        _hypervolume_contribution(a, [x for x in arch_objs if x != a])
        for a in arch_objs
    ]
    ranked = sorted(range(len(arch_objs)), key=lambda i: -contributions[i])
    return [pareto[i] for i in ranked[:k]]


# ---------------------------------------------------------------------------
# Deployment experiment
# ---------------------------------------------------------------------------

def deploy_and_measure(
    arch_ids: List[int],
    nasbench_path: str,
    dataset: str,
    output_dir: str,
    label: str,
) -> List[Dict]:
    """Measure latency, power, and memory for a list of arch_ids on Jetson Thor.

    Parameters
    ----------
    arch_ids :
        Architecture indices to profile.
    nasbench_path :
        Path to NATS-Bench TSS file (for decoding operation vectors).
    dataset :
        Dataset used in NAS ('cifar10', 'cifar100', 'ImageNet16-120').
    output_dir :
        Directory to save individual measurement JSON files.
    label :
        Descriptive label (e.g. 'nsga2_gated_cifar10').
    """
    loader  = NATSBenchLoader(nasbench_path, dataset=dataset)
    num_cls = {"cifar10": 10, "cifar100": 100, "ImageNet16-120": 120}[dataset]

    results = []
    for arch_id in arch_ids:
        ops = loader.arch_id_to_ops(arch_id)
        print(f"  Measuring arch_id={arch_id}  ops={ops}")

        result = measure_architecture(
            arch_id=arch_id,
            ops=ops,
            num_classes=num_cls,
            n_warmup=50,
            n_runs=1000,
        )
        entry = {
            "arch_id":         arch_id,
            "ops":             ops,
            "latency_ms":      round(result.mean_latency_ms, 3),
            "latency_cv":      round(result.cv_percent, 3),
            "power_w":         round(result.mean_power_w, 2),
            "energy_mj":       round(result.mean_latency_ms * result.mean_power_w / 1000, 4),
            "peak_memory_mb":  round(result.peak_memory_mb, 1),
        }
        results.append(entry)

        # Save per-arch result immediately (fault-tolerant)
        arch_file = os.path.join(output_dir, f"{label}_arch{arch_id}.json")
        with open(arch_file, "w") as f:
            json.dump(entry, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Hardware deployment experiments.")
    p.add_argument("--results_dir",   default="results/")
    p.add_argument("--paradigms",     nargs="+",
                   default=["nsga2", "enas", "lamoo", "random"])
    p.add_argument("--dataset",       default="cifar10")
    p.add_argument("--top_k",         type=int, default=3)
    p.add_argument("--nasbench_path",
                   default="data/nasbench201/NATS-tss-v1_0-3ffb9.pickle.pbz2")
    p.add_argument("--output",        default="results/hw_deploy/")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    table4: List[Dict] = []

    for paradigm in args.paradigms:
        for condition in ("gated", "ungated"):
            # Load the corresponding run_nas.py output
            result_file = os.path.join(
                args.results_dir,
                f"{paradigm}_{args.dataset}_{condition}_seed0.json",
            )
            if not os.path.exists(result_file):
                print(f"[WARN] Not found: {result_file}  (skipping)")
                continue

            with open(result_file) as f:
                run_results = json.load(f)

            # Aggregate Pareto fronts across all trials
            all_pareto = []
            for trial in run_results:
                all_pareto.extend(trial.get("final_pareto", []))

            # Deduplicate by arch_id
            seen    = set()
            pareto  = []
            for entry in all_pareto:
                if entry["arch_id"] not in seen:
                    seen.add(entry["arch_id"])
                    pareto.append(entry)

            # Top-k by HV contribution
            top_k  = top_k_by_hv_contribution(pareto, k=args.top_k)
            ids_to_measure = [e["arch_id"] for e in top_k]

            print(f"\n[DEPLOY] {paradigm}/{condition} — measuring {ids_to_measure}")
            label   = f"{paradigm}_{condition}_{args.dataset}"
            measurements = deploy_and_measure(
                arch_ids=ids_to_measure,
                nasbench_path=args.nasbench_path,
                dataset=args.dataset,
                output_dir=args.output,
                label=label,
            )

            for m in measurements:
                table4.append({
                    "paradigm":  paradigm,
                    "condition": condition,
                    "dataset":   args.dataset,
                    **m,
                })

    # Save aggregated Table 4
    out_file = os.path.join(args.output, f"table4_{args.dataset}.json")
    with open(out_file, "w") as f:
        json.dump(table4, f, indent=2)
    print(f"\n[INFO] Table 4 data saved to {out_file}")

    # Print summary
    print("\n{'paradigm':12s} {'condition':10s} {'lat_ms':8s} "
          "{'cv%':6s} {'pwr_w':7s} {'mj':8s}")
    for row in table4:
        print(
            f"{row['paradigm']:12s} {row['condition']:10s} "
            f"{row['latency_ms']:8.3f} {row['latency_cv']:6.2f} "
            f"{row['power_w']:7.2f} {row['energy_mj']:8.4f}"
        )


if __name__ == "__main__":
    main()
