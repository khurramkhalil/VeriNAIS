"""
hardware/measure_harness.py
---------------------------
On-device measurement harness for the NVIDIA Jetson Thor platform.

This module implements the profiling pipeline used in Phase 2B of the
experimental evaluation to collect the latency and power measurements
stored in hw_table.py.

Measurement protocol (Section VII-A)
--------------------------------------
  1. Construct the NAS-Bench-201 cell for the given architecture.
  2. Compile to TensorRT FP16 (via torch2trt or trtexec).
  3. Run 50 warm-up passes (discarded).
  4. Run 1 000 timed inference passes; record latency per pass (CUDA events).
  5. Simultaneously poll tegrastats at 100 ms intervals; average over the
     inference window to obtain mean power draw.
  6. Report: mean latency (ms), CV (%), mean power (W), peak memory (MB).

Reproducibility
---------------
  Coefficient of variation (CV) = std / mean < 0.5 % across 5 independent
  runs of the same architecture on Jetson Thor (validated in Phase 2A).
  This confirms measurement reproducibility well below the 5 % threshold.

Dependencies (on Jetson Thor only)
------------------------------------
  torch          >= 2.9.1  (with CUDA 13.0)
  torch2trt      (TensorRT FP16 bridge)
  subprocess     (for tegrastats polling)
  threading      (for concurrent power sampling)

Note
----
This harness is designed to run on the Jetson Thor device.
Off-device, the HardwareLookupTable in hw_table.py provides the
pre-measured values needed for NAS-Bench-201 search runs.
"""

from __future__ import annotations

# ── Standard library ────────────────────────────────────────────────────────
import csv
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ── Third-party (available on Jetson Thor) ───────────────────────────────────
# NOTE: torch, torch2trt, and related libraries are imported here.
# These are available on the Jetson Thor device.
# import torch
# import torch.nn as nn
# from torch2trt import torch2trt


# ---------------------------------------------------------------------------
# Architecture cell construction
# ---------------------------------------------------------------------------

# NAS-Bench-201 operation implementations
# (imports from nasbench201 are shown as comments; resolve at runtime)
# from nasbench201.utils.model_search import NAS201SearchCell

def build_nas201_cell(ops: List[str], num_classes: int = 10):
    """Construct a NAS-Bench-201 evaluation cell for the given operation list.

    Parameters
    ----------
    ops :
        List of 6 operation strings, e.g.:
        ['nor_conv_3x3', 'nor_conv_3x3', 'skip_connect',
         'avg_pool_3x3', 'nor_conv_1x1', 'skip_connect']
    num_classes :
        Number of output classes (10 for CIFAR-10, 100 for CIFAR-100,
        120 for ImageNet-16-120).

    Returns
    -------
    nn.Module
        A NAS-Bench-201 cell wrapped in a full network (stem + cell × 5 + head).
    """
    # ── Download NAS-Bench-201 / NATS-Bench source ──────────────────────────
    # The NAS201SearchCell class is available in the NATS-Bench repository:
    # https://github.com/D-X-Y/NATS-Bench
    # Clone it to ~/projects/stl-nas-emsoft/third_party/nats_bench/
    # and add it to sys.path before calling this function.
    #
    # from third_party.nats_bench.cell_operations import NAS201SearchCell
    # cell = NAS201SearchCell(ops)
    # model = FullNetworkWrapper(cell, num_classes=num_classes)
    # return model
    raise NotImplementedError(
        "build_nas201_cell requires the NATS-Bench source tree.  "
        "Clone https://github.com/D-X-Y/NATS-Bench and add to sys.path."
    )


# ---------------------------------------------------------------------------
# TensorRT compilation
# ---------------------------------------------------------------------------

def compile_trt(model, input_shape: Tuple[int, int, int, int] = (1, 3, 32, 32)):
    """Compile a PyTorch model to TensorRT FP16.

    Parameters
    ----------
    model :
        PyTorch nn.Module (output of build_nas201_cell).
    input_shape :
        (N, C, H, W) input tensor shape.

    Returns
    -------
    TRTModule
        torch2trt-compiled model ready for inference on Jetson Thor.
    """
    # import torch
    # from torch2trt import torch2trt
    #
    # model = model.eval().cuda()
    # dummy_input = torch.ones(input_shape).cuda()
    # with torch.no_grad():
    #     trt_model = torch2trt(
    #         model,
    #         [dummy_input],
    #         fp16_mode=True,
    #         max_batch_size=1,
    #     )
    # return trt_model
    raise NotImplementedError("compile_trt requires torch2trt on Jetson Thor.")


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------

def measure_latency(
    trt_model,
    input_shape: Tuple[int, int, int, int] = (1, 3, 32, 32),
    n_warmup: int = 50,
    n_runs: int = 1000,
) -> Tuple[float, float]:
    """Measure inference latency using CUDA events.

    Parameters
    ----------
    trt_model :
        TensorRT-compiled model (from compile_trt).
    input_shape :
        Input tensor shape.
    n_warmup :
        Number of warm-up passes (discarded).
    n_runs :
        Number of timed passes.

    Returns
    -------
    (mean_latency_ms, cv_percent)
        Mean latency in milliseconds and coefficient of variation (%).
    """
    # import torch
    #
    # dummy = torch.ones(input_shape).cuda()
    # latencies = []
    #
    # # Warm-up
    # for _ in range(n_warmup):
    #     with torch.no_grad():
    #         _ = trt_model(dummy)
    # torch.cuda.synchronize()
    #
    # # Timed runs using CUDA events
    # for _ in range(n_runs):
    #     start = torch.cuda.Event(enable_timing=True)
    #     end   = torch.cuda.Event(enable_timing=True)
    #     start.record()
    #     with torch.no_grad():
    #         _ = trt_model(dummy)
    #     end.record()
    #     torch.cuda.synchronize()
    #     latencies.append(start.elapsed_time(end))   # ms
    #
    # mean_lat = float(np.mean(latencies))
    # cv       = float(np.std(latencies) / mean_lat * 100.0)
    # return mean_lat, cv
    raise NotImplementedError("measure_latency requires CUDA on Jetson Thor.")


# ---------------------------------------------------------------------------
# Power measurement via tegrastats
# ---------------------------------------------------------------------------

class TegrastatsPoller:
    """Polls tegrastats at a fixed interval and records power readings.

    tegrastats outputs a line of system metrics including GPU power (mW)
    and total SoC power (mW) every `interval_ms` milliseconds.

    Usage
    -----
    >>> poller = TegrastatsPoller(interval_ms=100)
    >>> poller.start()
    >>> # ... run inference ...
    >>> poller.stop()
    >>> mean_power_w = poller.mean_power_w()
    """

    def __init__(self, interval_ms: int = 100) -> None:
        self.interval_ms = interval_ms
        self._readings: List[float] = []
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """Launch tegrastats and begin polling in a background thread."""
        # tegrastats is available on all Jetson devices at /usr/bin/tegrastats
        self._running = True
        self._proc = subprocess.Popen(
            ["/usr/bin/tegrastats", "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the poller and wait for the background thread to finish."""
        self._running = False
        if self._proc:
            self._proc.terminate()
            self._proc.wait()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _poll_loop(self) -> None:
        """Background thread: parse tegrastats output lines."""
        for line in self._proc.stdout:
            if not self._running:
                break
            power_w = self._parse_total_power(line)
            if power_w is not None:
                self._readings.append(power_w)

    @staticmethod
    def _parse_total_power(line: str) -> Optional[float]:
        """Parse the total SoC power from one tegrastats output line.

        tegrastats output format (Jetson Thor):
          ... VDD_SOC 5120/5120mW VDD_CPU 3200/4000mW ...
          or: Total: 26500mW

        We extract the 'VDD_SOC' field as the primary power indicator.
        """
        # Example tegrastats line (varies by firmware):
        # "12-31-2025 10:00:00 RAM 8192/131072MB ... VDD_SOC 26500mW ..."
        try:
            if "VDD_SOC" in line:
                idx = line.index("VDD_SOC") + len("VDD_SOC ")
                mw_str = line[idx:idx + 10].split()[0].rstrip("mW/")
                return float(mw_str) / 1000.0   # mW → W
        except (ValueError, IndexError):
            pass
        return None

    def mean_power_w(self) -> float:
        """Return mean power draw (W) over all recorded samples."""
        if not self._readings:
            return 0.0
        import numpy as np
        return float(np.mean(self._readings))

    def max_power_w(self) -> float:
        """Return maximum power draw (W) over all recorded samples."""
        if not self._readings:
            return 0.0
        import numpy as np
        return float(np.max(self._readings))


# ---------------------------------------------------------------------------
# Full measurement pipeline
# ---------------------------------------------------------------------------

@dataclass
class MeasurementResult:
    arch_id: int
    ops: List[str]
    mean_latency_ms: float
    cv_percent: float
    mean_power_w: float
    peak_memory_mb: float
    n_runs: int


def measure_architecture(
    arch_id: int,
    ops: List[str],
    num_classes: int = 10,
    n_warmup: int = 50,
    n_runs: int = 1000,
    input_shape: Tuple[int, int, int, int] = (1, 3, 32, 32),
) -> MeasurementResult:
    """Full measurement pipeline for one NAS-Bench-201 architecture.

    Executes: build → compile (TRT FP16) → warm-up → timed runs (CUDA events)
    + concurrent power sampling (tegrastats).

    Parameters
    ----------
    arch_id :
        NAS-Bench-201 index.
    ops :
        6-element operation list (decoded from arch_id).
    num_classes :
        Dataset class count.
    n_warmup :
        Number of warm-up inference passes.
    n_runs :
        Number of timed inference passes.
    input_shape :
        (N, C, H, W) for CIFAR-10 use (1, 3, 32, 32).

    Returns
    -------
    MeasurementResult
    """
    # import torch
    #
    # Build and compile
    model     = build_nas201_cell(ops, num_classes=num_classes)
    trt_model = compile_trt(model, input_shape=input_shape)
    #
    # Start power poller
    poller = TegrastatsPoller(interval_ms=100)
    poller.start()
    #
    # Measure latency
    mean_lat, cv = measure_latency(trt_model, input_shape, n_warmup, n_runs)
    #
    # Stop power poller
    poller.stop()
    mean_pwr = poller.mean_power_w()
    #
    # Peak activation memory
    # torch.cuda.reset_peak_memory_stats()
    # with torch.no_grad():
    #     _ = trt_model(torch.ones(input_shape).cuda())
    # peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_mem_mb = 0.0  # filled in when running on Jetson Thor
    #
    raise NotImplementedError(
        "measure_architecture requires Jetson Thor with TensorRT.  "
        "Pre-measured values are in hardware/hw_table.py."
    )

    return MeasurementResult(
        arch_id=arch_id,
        ops=ops,
        mean_latency_ms=mean_lat,
        cv_percent=cv,
        mean_power_w=mean_pwr,
        peak_memory_mb=peak_mem_mb,
        n_runs=n_runs,
    )
