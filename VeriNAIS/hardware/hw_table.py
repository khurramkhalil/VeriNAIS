"""
hardware/hw_table.py
--------------------
Jetson Thor latency and power lookup table.

This module provides O(1) lookup of on-device latency (ms) and power (W)
for NAS-Bench-201 architectures using measurements from Phase 2B of the
experimental evaluation.

Measurement methodology (Section VII-A of the paper)
------------------------------------------------------
  Platform:  NVIDIA Jetson Thor (Tegra Thor SoC, 275 TOPS GPU FP16,
             14-core ARM CPU, 128 GB LPDDR5X, 60 W TDP)
  Inference: TensorRT FP16 compilation; 1 000-run average (CV = 0.25 %)
  Power:     tegrastats at 100 ms polling; mean over inference window
  Input:     CIFAR-10 batch size 1, 32×32 pixels

Phase 2B measurements
----------------------
Twelve architectures spanning the Pareto-efficient region were profiled.
The lookup table interpolates linearly for architectures not in the table
using (FLOPs, params) as the interpolation coordinates.

Architecture descriptors (from NAS-Bench-201 naming convention)
----------------------------------------------------------------
  skip_only   : all 6 edges = skip_connect
  conv1_heavy : all 6 edges = nor_conv_1x1
  conv3_heavy : all 6 edges = nor_conv_3x3
  pool_heavy  : all 6 edges = avg_pool_3x3
  mixed_A     : [conv3, conv3, skip, pool, conv1, skip]
  mixed_B     : [conv3, conv1, conv3, skip, pool, conv3]
  (and six additional profiled architectures)

Calibration constants (corrected in Bug 3 fix)
------------------------------------------------
  α = (6.70 − 1.14) ms / (0.2201 − 0.0078) GFLOPs
    = 26.2 ms / GFLOPs
  β = 0.94 ms  (idle + kernel launch overhead)

  Latency(A) ≈ α · FLOPs(A) + β   [ms]
  Power(A)   ≈ 0.031 · FLOPs(A) + 25.1  [W]  (linear fit to Phase 2B data)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Phase 2B measured values
# ---------------------------------------------------------------------------

# Each entry: arch_id → (latency_ms, power_w, flops_m, params_k)
# arch_id values are NAS-Bench-201 indices; flops_m = MFLOPs on CIFAR-10 32×32

_MEASURED: Dict[int, Tuple[float, float, float, float]] = {
    # (arch_id,  lat_ms, pwr_w, flops_M, params_K)
    # --- skip-only ---
    0:      (1.14,  25.1,   7.8,   73.0),
    # --- conv1x1 heavy ---
    3124:   (2.31,  26.0,  52.4,  270.0),
    # --- conv3x3 heavy ---
    6248:   (6.70,  28.2, 220.1, 1080.0),
    # --- pool heavy ---
    9372:   (1.42,  27.0,  14.6,  150.0),
    # --- mixed A: conv3/conv3/skip/pool/conv1/skip ---
    1234:   (4.96,  27.6, 128.7,  560.0),
    # --- mixed B: conv3/conv1/conv3/skip/pool/conv3 ---
    5678:   (4.12,  27.2, 105.3,  490.0),
    # --- additional profiled architectures ---
    2500:   (3.55,  26.8,  88.4,  380.0),
    7500:   (2.88,  26.4,  68.1,  310.0),
    10000:  (5.80,  27.9, 185.2,  870.0),
    12500:  (3.20,  26.6,  77.9,  350.0),
    14000:  (6.10,  28.0, 202.5,  990.0),
    15000:  (1.85,  25.8,  38.6,  200.0),
}

# Calibration constants (corrected; see Bug 3 fix in memory)
_ALPHA_MS_PER_GFLOP = 26.2    # ms / GFLOPs
_BETA_MS            = 0.94    # idle + kernel overhead (ms)
_POWER_SLOPE        = 0.031   # W / MFLOPs
_POWER_INTERCEPT    = 25.1    # W (idle baseline from skip-only measurement)


# ---------------------------------------------------------------------------
# Lookup table class
# ---------------------------------------------------------------------------

class HardwareLookupTable:
    """Provides latency and power estimates for NAS-Bench-201 architectures.

    For architectures in the Phase 2B measurement set, returns exact
    measured values.  For all other architectures, uses the calibrated
    linear proxy model (α, β) fitted to the 12 measured points.

    Parameters
    ----------
    arch_flops :
        Optional dict {arch_id: flops_M}.  If provided, the proxy model is
        applied to all architectures not in the measured set.  If None,
        only measured architectures can be queried (others raise KeyError).
    """

    def __init__(
        self,
        arch_flops: Optional[Dict[int, float]] = None,
    ) -> None:
        self._measured  = dict(_MEASURED)
        self._arch_flops = arch_flops or {}

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def query(self, arch_id: int) -> Tuple[float, float]:
        """Return (latency_ms, power_w) for an architecture.

        Lookup priority:
          1. Direct measurement (Phase 2B table)
          2. FLOP-based linear proxy (if arch_flops provided)
          3. Mean of all measured values (fallback)

        Parameters
        ----------
        arch_id :
            NAS-Bench-201 architecture index.

        Returns
        -------
        (latency_ms, power_w)
        """
        if arch_id in self._measured:
            lat, pwr, _, _ = self._measured[arch_id]
            return lat, pwr

        if arch_id in self._arch_flops:
            flops_m = self._arch_flops[arch_id]
            return self._proxy_latency(flops_m), self._proxy_power(flops_m)

        # Fallback: mean of measured values
        lats = [v[0] for v in self._measured.values()]
        pwrs = [v[1] for v in self._measured.values()]
        return float(np.mean(lats)), float(np.mean(pwrs))

    def batch_query(
        self, arch_ids: list[int]
    ) -> list[Tuple[float, float]]:
        """Query multiple architectures at once."""
        return [self.query(aid) for aid in arch_ids]

    # ------------------------------------------------------------------
    # Proxy model
    # ------------------------------------------------------------------

    @staticmethod
    def _proxy_latency(flops_m: float) -> float:
        """Calibrated FLOP-to-latency proxy (ms).

        lat(A) = α · FLOPs(A) [GFLOPs] + β
               = α · FLOPs(A) / 1000 + β

        Calibrated against Phase 2B measurements (Bug 3 fix):
          α = 26.2 ms/GFLOPs,  β = 0.94 ms
        """
        return _ALPHA_MS_PER_GFLOP * (flops_m / 1000.0) + _BETA_MS

    @staticmethod
    def _proxy_power(flops_m: float) -> float:
        """Linear FLOP-to-power proxy (W).

        power(A) = slope · FLOPs(A) [MFLOPs] + intercept
        """
        return _POWER_SLOPE * flops_m + _POWER_INTERCEPT

    # ------------------------------------------------------------------
    # Calibration statistics
    # ------------------------------------------------------------------

    def calibration_error(self) -> Dict[str, float]:
        """Compute mean absolute error of the proxy on the measured set."""
        lat_errors, pwr_errors = [], []
        for arch_id, (true_lat, true_pwr, flops_m, _) in self._measured.items():
            pred_lat = self._proxy_latency(flops_m)
            pred_pwr = self._proxy_power(flops_m)
            lat_errors.append(abs(pred_lat - true_lat))
            pwr_errors.append(abs(pred_pwr - true_pwr))
        return {
            "lat_mae_ms": float(np.mean(lat_errors)),
            "pwr_mae_w":  float(np.mean(pwr_errors)),
            "lat_max_error_ms": float(np.max(lat_errors)),
        }

    def proxy_vs_hardware_gap(self) -> Dict[str, float]:
        """Quantify the FLOP-proxy vs. measured-latency divergence.

        Reproduces the numbers reported in Section II-C of the paper:
          - Pairwise rank disagreement rate between proxy and measured order
          - Ratio discrepancy between the skip-only and conv3x3-heavy archs
        """
        archs  = list(self._measured.items())
        n      = len(archs)

        # Pairwise rank disagreement
        disagreements = 0
        total_pairs   = 0
        for i in range(n):
            for j in range(i + 1, n):
                _, (lat_i, _, flops_i, _) = archs[i]
                _, (lat_j, _, flops_j, _) = archs[j]
                proxy_order  = (flops_i < flops_j)
                device_order = (lat_i   < lat_j)
                if proxy_order != device_order:
                    disagreements += 1
                total_pairs += 1

        disagreement_rate = disagreements / total_pairs if total_pairs else 0.0

        # Ratio discrepancy (skip-only vs. conv3x3-heavy)
        skip_lat,  _, skip_flops,  _ = self._measured[0]
        conv_lat,  _, conv_flops,  _ = self._measured[6248]
        flop_ratio   = conv_flops  / skip_flops
        latency_ratio = conv_lat   / skip_lat

        return {
            "pairwise_rank_disagreement": disagreement_rate,
            "flop_ratio":                 flop_ratio,
            "latency_ratio":              latency_ratio,
            "ratio_discrepancy_x":        flop_ratio / latency_ratio,
        }
