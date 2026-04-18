"""
data/nasbench201.py
-------------------
NAS-Bench-201 / NATS-Bench TSS benchmark loader.

This module provides a lightweight wrapper around the NATS-Bench TSS API
that returns Architecture objects (as defined in stl_nas/signal.py) for
use by the NAS oracles and the STL gatekeeper.

Benchmark: NATS-Bench Topology Search Space (TSS)
--------------------------------------------------
  Search space:  15 625 architectures (5^6 operation combinations)
  Datasets:      CIFAR-10, CIFAR-100, ImageNet-16-120
  Accuracy:      Pre-computed at epoch 200 (full training)
  File:          NATS-tss-v1_0-3ffb9.pickle.pbz2 (1.08 GB)
  Download:      https://github.com/D-X-Y/NATS-Bench

Loading instructions
--------------------
  1. Download NATS-tss-v1_0-3ffb9.pickle.pbz2 from the NATS-Bench
     repository and place it at:
       data/nasbench201/NATS-tss-v1_0-3ffb9.pickle.pbz2

  2. Install the nats_bench package:
       pip install nats-bench

  3. Call NATSBenchLoader.load() to obtain the full architecture pool.

Usage
-----
  from data.nasbench201 import NATSBenchLoader

  loader = NATSBenchLoader(
      path="data/nasbench201/NATS-tss-v1_0-3ffb9.pickle.pbz2",
      dataset="cifar10",
  )
  arch_pool = loader.load()         # list of Architecture objects
  arch      = loader.query(arch_id) # single Architecture by index
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

# stl_nas imports
from stl_nas.signal import Architecture

# NATS-Bench API
# Download from: https://github.com/D-X-Y/NATS-Bench
# pip install nats-bench
# import nats_bench


# ---------------------------------------------------------------------------
# Operation constants (NAS-Bench-201)
# ---------------------------------------------------------------------------

OPS = [
    "none",
    "skip_connect",
    "nor_conv_1x1",
    "nor_conv_3x3",
    "avg_pool_3x3",
]
N_OPS   = 5
N_EDGES = 6

# FLOP counts per operation type on CIFAR-10 (32×32 input, one cell)
# Used for the energy proxy: energy(A) = FLOPs(A) / FLOPs_max
_FLOPS_PER_OP_M: Dict[str, float] = {
    "none":           0.0,
    "skip_connect":   0.13,
    "nor_conv_1x1":   7.3,
    "nor_conv_3x3":   36.3,
    "avg_pool_3x3":   2.4,
}
_FLOPS_MAX_M = 220.1   # conv3x3-heavy architecture on CIFAR-10 32×32

# Approximate parameter counts per operation (in the 5-cell network)
_PARAMS_PER_OP: Dict[str, int] = {
    "none":           0,
    "skip_connect":   0,
    "nor_conv_1x1":   45_000,
    "nor_conv_3x3":   180_000,
    "avg_pool_3x3":   0,
}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class NATSBenchLoader:
    """Loads the NATS-Bench TSS benchmark and converts entries to Architecture.

    Parameters
    ----------
    path :
        Absolute path to NATS-tss-v1_0-3ffb9.pickle.pbz2.
    dataset :
        One of 'cifar10', 'cifar100', 'ImageNet16-120'.
    hw_lookup :
        Optional HardwareLookupTable instance.  If provided, fills lat_ms
        and pwr_w from the lookup table; otherwise uses the proxy model.
    """

    VALID_DATASETS = ("cifar10", "cifar100", "ImageNet16-120")

    def __init__(
        self,
        path: str,
        dataset: str = "cifar10",
        hw_lookup=None,
    ) -> None:
        if dataset not in self.VALID_DATASETS:
            raise ValueError(
                f"dataset must be one of {self.VALID_DATASETS}, got '{dataset}'"
            )
        self.path       = path
        self.dataset    = dataset
        self.hw_lookup  = hw_lookup
        self._api       = None
        self._cache: Dict[int, Architecture] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> List[Architecture]:
        """Load all 15 625 architectures from the NATS-Bench TSS file.

        Returns a list of Architecture objects sorted by arch_id.

        Loading takes approximately 40 seconds and requires ~25-28 GB RAM.
        """
        # ── Download the NATS-Bench TSS file ──────────────────────────────
        # Place NATS-tss-v1_0-3ffb9.pickle.pbz2 at self.path.
        # Download from: https://github.com/D-X-Y/NATS-Bench
        #
        # self._api = nats_bench.create(
        #     str(self.path), 'tss',
        #     fast_mode=False, verbose=False
        # )
        # print(f"[NATSBench] Loaded {len(self._api)} architectures "
        #       f"for dataset '{self.dataset}'")
        #
        # archs = []
        # for arch_id in range(len(self._api)):
        #     arch = self._load_one(arch_id)
        #     archs.append(arch)
        #     self._cache[arch_id] = arch
        # return archs
        raise NotImplementedError(
            "Place NATS-tss-v1_0-3ffb9.pickle.pbz2 at the configured path "
            "and install nats_bench (pip install nats-bench) to load the benchmark."
        )

    def query(self, arch_id: int) -> Architecture:
        """Retrieve one Architecture by its NAS-Bench-201 index.

        Parameters
        ----------
        arch_id :
            Index in [0, 15 624].
        """
        if arch_id in self._cache:
            return self._cache[arch_id]
        if self._api is None:
            raise RuntimeError("Call load() before query().")
        arch = self._load_one(arch_id)
        self._cache[arch_id] = arch
        return arch

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_one(self, arch_id: int) -> Architecture:
        """Convert one NATS-Bench entry to an Architecture object."""
        # ── Query accuracy from benchmark table ──────────────────────────
        # result = self._api.get_more_info(arch_id, self.dataset,
        #                                  hp='200', is_random=False)
        # acc = result['test-accuracy'] / 100.0  # normalise to [0, 1]
        #
        # ── Decode operation vector from arch string ──────────────────────
        # arch_str = self._api.arch(arch_id)
        # ops = self._decode_ops(arch_str)
        #
        # ── Compute energy proxy ─────────────────────────────────────────
        # flops_m = sum(_FLOPS_PER_OP_M[op] for op in ops)
        # energy  = min(flops_m / _FLOPS_MAX_M, 2.0)
        #
        # ── Compute parameter count ──────────────────────────────────────
        # params  = sum(_PARAMS_PER_OP[op] for op in ops)
        #
        # ── Conv-op fraction ─────────────────────────────────────────────
        # conv_ops = {'nor_conv_1x1', 'nor_conv_3x3'}
        # conv_fraction = sum(1 for op in ops if op in conv_ops) / N_EDGES
        #
        # ── Latency / power from hardware lookup ─────────────────────────
        # if self.hw_lookup is not None:
        #     lat_ms, pwr_w = self.hw_lookup.query(arch_id)
        # else:
        #     from hardware.hw_table import HardwareLookupTable
        #     proxy = HardwareLookupTable()
        #     lat_ms, pwr_w = proxy.query(arch_id)
        #
        # return Architecture(
        #     arch_id=arch_id,
        #     acc=acc * 100.0,   # store as percentage for STL signal
        #     energy=energy,
        #     params=params,
        #     conv_fraction=conv_fraction,
        #     lat_ms=lat_ms,
        #     pwr_w=pwr_w,
        # )
        raise NotImplementedError(
            "NATS-Bench API not loaded.  Call NATSBenchLoader.load() first."
        )

    @staticmethod
    def _decode_ops(arch_str: str) -> List[str]:
        """Decode a NAS-Bench-201 architecture string to a list of op names.

        Architecture string format (example):
          '|nor_conv_3x3~0|+|nor_conv_3x3~0|skip_connect~1|+'
          '|avg_pool_3x3~0|nor_conv_1x1~1|nor_conv_1x1~2|'

        Returns a flat list of 6 operation names in edge order.
        """
        ops = []
        for token in arch_str.split("|"):
            token = token.strip()
            if "~" in token:
                op = token.split("~")[0]
                ops.append(op)
        return ops[:N_EDGES]

    @staticmethod
    def arch_id_to_ops(arch_id: int) -> List[str]:
        """Decode arch_id to operation names using base-5 encoding."""
        ops = []
        n   = arch_id
        for _ in range(N_EDGES):
            ops.append(OPS[n % N_OPS])
            n //= N_OPS
        return ops
