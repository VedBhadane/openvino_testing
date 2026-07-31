"""
hardware_benchmark_flow.py
==========================

A self-contained ML workload used to BENCHMARK SERVER HARDWARE.

Why this file exists
--------------------
Intel has given us a pre-production "Wildcat Lake" (Core 300 / Core Series 3) machine
and, as part of our partnership, wants us to compare it against our existing data-center
server (the "baseline"). They sent the comparison table below. The table is generic; this
script is our concrete answer to it. We run THIS SAME script on both machines and compare
the numbers it prints. This specific file intends to compare the performance of Intel's 
OpenVINO Runtime (CPU plugin) on the two machines, using a small CNN workload.

What the Intel table actually means (their image, decoded)
----------------------------------------------------------
The table has one row per "thing to report" and columns: Baseline system | NEW system | Delta %.

    Row                       Meaning for us
    ------------------------  --------------------------------------------------------------
    CPU / GPU / Memory        Hardware spec of each machine. We auto-print these (see the
                              SYSTEM INFO block at the top of the output). "NEW = Wildcat
                              Lake", "Baseline = our current server". GPU = N/A (we have no
                              GPU and this workload is CPU-only — Intel can ignore that row).
    SW workload               WHICH program we ran to get the numbers = this script. One line.
    Inference engine & version  The runtime doing the math = OpenVINO Runtime (CPU plugin).
                              Version is auto-printed. Training still happens in PyTorch
                              (autograd is needed for that), but the model is converted to an
                              OpenVINO IR immediately after training and BOTH inference KPIs
                              (throughput + latency) are measured through the OpenVINO
                              compiled model, not raw PyTorch eager mode. This matters because
                              OpenVINO applies Intel-tuned CPU kernels (oneDNN/oneAPI graph
                              fusion, AVX/AMX code paths) that PyTorch eager mode does not use
                              by default — so this is the fairer "how fast can this CPU really
                              serve this model" number for Intel's table.
    KPI 1  Throughput         How MUCH work per second. Here: inference samples processed
                              per second under sustained load, via the OpenVINO compiled
                              model. Higher = better.
    KPI 2  Latency            How LONG a single request takes. Here: time for one
                              single-sample OpenVINO inference, in milliseconds. Lower = better.
    Delta %                   The single headline number Intel asked for: how much better
                              (or worse) the NEW machine is than the baseline, combining all
                              three KPIs. See "The single Delta %" below.

How we use it (the procedure)
-----------------------------
1. Run `python hardware_benchmark_flow.py` on the BASELINE server. Note the printed
   "COMPOSITE SCORE S".
2. Run the exact same command on the Wildcat Lake machine.
3. On the Wildcat Lake run, pass the baseline's score so it prints the Delta % directly:
   set BASELINE_SCORE in __main__ (or call func(baseline_score=<baseline S>)).
   (Keep `num_runs`, problem sizes and core/thread count identical on both machines, otherwise
   the comparison is meaningless. Memory, core count and disk are the same per the agreement;
   only the CPU model differs — which is exactly what we are measuring.)

Why 5 runs
----------
A single timing is noisy (cache state, OS scheduling, thermal). We run the whole workload
`num_runs` times (default 5), print every individual run's KPIs, then average them. The
average is what feeds the Delta %.

The single Delta %  (the one number Intel wants)
------------------------------------------------
Each KPI has a direction (throughput up = good; latency down = good; power down = good).
We fold all three into ONE machine score using a geometric mean so the units cancel:

        S = ( Throughput / (Latency_ms * Power_W) ) ** (1/3)

S is "higher is better" and is unitless-by-ratio (its absolute value is meaningless; only
the ratio between two machines matters). The headline delta is then simply:

        Delta % = (S_new / S_baseline - 1) * 100

Because of how S is built, this single number is mathematically identical to the geometric
mean of the three individual KPI improvements:

        Delta % = ( cbrt( (T_new/T_base) * (L_base/L_new) * (P_base/P_new) ) - 1 ) * 100

So a positive Delta % means Wildcat Lake is better overall; negative means worse. We also
print the per-KPI deltas next to it so Intel can see where the gain/loss comes from.

Notes / environment
-------------------
- Runs fully OFFLINE. No dataset download, no internet. The data is synthetic and seeded,
  so both machines train/infer on byte-identical inputs and do byte-identical work.
- Pure CPU. We never touch a GPU. OpenVINO is compiled/run with the "CPU" device plugin only.
- Power is read from Intel RAPL (`/sys/class/powercap/intel-rapl:*/energy_uj`). Reading it
  often needs root (the files can be 0400). If RAPL is unreadable the script still runs and
  reports all timing KPIs, but Power shows as unavailable and the composite score falls back
  to throughput+latency only (this is clearly flagged in the output — tell Intel which mode
  you ran in). To get the power KPI, run as root / with read access to the RAPL sysfs files.
- Libraries assumed present in the target environment: torch, numpy, openvino. psutil is
  optional (only used to enrich the system-info banner). The model->IR conversion uses
  `openvino.convert_model`, which ships as part of the `openvino` package (no separate
  `openvino-dev`/mo install needed on recent OpenVINO versions).
"""

import os
import re
import glob
import json
import time
import math
import platform
import statistics

print("[BENCH] hardware_benchmark_flow.py: file loaded, importing numpy/torch/openvino "
      "(these imports can take ~5-15s the first time) ...", flush=True)
import numpy as np
import torch
import torch.nn as nn
import openvino as ov
print(f"[BENCH] imports OK — numpy {np.__version__}, torch {torch.__version__}, "
      f"openvino {ov.__version__}", flush=True)


# ----------------------------------------------------------------------------------------
# 0. DEBUG LOGGING
# ----------------------------------------------------------------------------------------
# This is a throwaway benchmark script, so we log loudly. Every _dbg() line is timestamped
# (seconds since the benchmark started) and flushed immediately, so if a run hangs or a CPU
# is crawling you can see exactly which phase/batch it's stuck on in real time (important when
# the same script is run over SSH on the Wildcat Lake box). Toggle with the `debug` arg to func.

_DEBUG = True            # set False (or pass debug=False to func) to silence the [DBG ...] lines
_T0 = time.perf_counter()  # benchmark wall-clock origin; reset at the start of func()


def _emit(msg):
    """Emit one log line.

    ALWAYS prints to stdout with flush=True — that's what the Airflow task log captures, so
    this is the output you see when tailing the run. (Earlier this routed through the engine's
    injected `_logger` *first*, which sends lines to the SSE/json_logs stream instead of the
    Airflow stdout log — making the run look silent/hung. Don't do that again.) If `_logger`
    is present we ALSO forward to it, so the SSE stream gets the lines too — but stdout is the
    source of truth.
    """
    print(msg, flush=True)
    fn = globals().get("_logger")
    if callable(fn):
        try:
            fn(msg)
        except Exception:
            pass


def _dbg(msg):
    """Emit a timestamped debug line (gated by _DEBUG)."""
    if _DEBUG:
        _emit(f"[DBG +{time.perf_counter() - _T0:7.2f}s] {msg}")


def _out(msg=""):
    """Emit a normal (user-facing) report line."""
    _emit(msg)


# ----------------------------------------------------------------------------------------
# 1. POWER MEASUREMENT  (Intel RAPL)
# ----------------------------------------------------------------------------------------
# RAPL ("Running Average Power Limit") exposes a monotonically increasing energy counter, in
# microjoules, per CPU package, under /sys/class/powercap/. Average power over a window is just
# (energy_after - energy_before) / elapsed_seconds. The counter wraps around at
# `max_energy_range_uj`, so we correct for that. We sum across all top-level packages
# (intel-rapl:0, intel-rapl:1, ...) and IGNORE the sub-zones (intel-rapl:0:0 = "core",
# intel-rapl:0:1 = "uncore") so we don't double-count.

# def _rapl_package_files():
#     """Return [(energy_uj_path, max_energy_range_uj), ...] for each top-level package, or []."""
#     packages = []
#     # Match only top-level packages like 'intel-rapl:0', never sub-zones 'intel-rapl:0:1'.
#     candidates = sorted(glob.glob("/sys/class/powercap/intel-rapl:*/energy_uj"))
#     _dbg(f"RAPL: scanning powercap, found {len(candidates)} energy_uj file(s)")
#     for energy_path in candidates:
#         parent = os.path.dirname(energy_path)
#         if not re.match(r"^intel-rapl:\d+$", os.path.basename(parent)):
#             _dbg(f"RAPL:   skip sub-zone {energy_path}")
#             continue
#         try:
#             with open(energy_path) as f:
#                 f.read()  # probe readability up front; raises if we lack permission
#             max_path = os.path.join(parent, "max_energy_range_uj")
#             max_range = int(open(max_path).read().strip()) if os.path.exists(max_path) else None
#             packages.append((energy_path, max_range))
#             _dbg(f"RAPL:   usable package {energy_path} (max_range={max_range})")
#         except (OSError, ValueError) as e:
#             # Unreadable (permissions) or malformed — skip; we'll fall back gracefully.
#             _dbg(f"RAPL:   UNREADABLE {energy_path} ({e.__class__.__name__}) -> need root?")
#             continue
#     _dbg(f"RAPL: {len(packages)} usable package(s) -> power KPI {'ON' if packages else 'OFF (fallback)'}")
#     return packages


# def _read_energy_uj(packages):
#     """Sum the current energy counters (microjoules) across the given packages."""
#     total = 0
#     for energy_path, _ in packages:
#         total += int(open(energy_path).read().strip())
#     return total


# class _PowerMeter:
#     """Context manager: measures average CPU-package power (watts) over the wrapped block.

#     Usage:
#         meter = _PowerMeter()
#         with meter:
#             ...do work...
#         meter.avg_watts   # None if RAPL was unavailable
#     """

#     def __init__(self):
#         self._packages = _rapl_package_files()
#         self.available = len(self._packages) > 0
#         self.avg_watts = None
#         self.energy_joules = None
#         self._e0 = None
#         self._t0 = None

#     def __enter__(self):
#         if self.available:
#             self._e0 = _read_energy_uj(self._packages)
#             self._t0 = time.perf_counter()
#         return self

#     def __exit__(self, *exc):
#         if not self.available:
#             return False
#         if self._t0 is None or self._e0 is None:
#             return False
#         dt = time.perf_counter() - self._t0
#         e1 = _read_energy_uj(self._packages)
#         delta_uj = e1 - self._e0
#         if delta_uj < 0:
#             # Counter wrapped around — add each package's range once (best effort).
#             _dbg(f"RAPL: counter wrap detected (delta={delta_uj} uj), correcting")
#             for _, max_range in self._packages:
#                 if max_range:
#                     delta_uj += max_range
#         self.energy_joules = delta_uj / 1e6
#         self.avg_watts = (self.energy_joules / dt) if dt > 0 else None
#         _dbg(f"RAPL: measured {self.energy_joules:.2f} J over {dt:.2f}s -> {self.avg_watts:.2f} W")
#         return False


# ----------------------------------------------------------------------------------------
# 2. SYSTEM INFO  (auto-fills the CPU / Memory / engine rows of the Intel table)
# ----------------------------------------------------------------------------------------
def _cpu_model():
    """Best-effort human-readable CPU name (no internet, just /proc/cpuinfo / platform)."""
    try:
        for line in open("/proc/cpuinfo"):
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _total_memory_gb():
    try:
        import psutil
        return round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        return None


def _openvino_cpu_device_info(core):
    """Best-effort OpenVINO CPU plugin identification string (device name Core sees)."""
    try:
        return core.get_property("CPU", "FULL_DEVICE_NAME")
    except Exception:
        return "unknown"


def print_system_info(core):
    """Print the hardware/software banner that feeds the top rows of Intel's table."""
    _out("=" * 78)
    _out("SYSTEM INFO  (fills the CPU / Memory / SW workload / inference-engine rows)")
    _out("-" * 78)
    _out(f"  CPU model           : {_cpu_model()}")
    _out(f"  OpenVINO sees CPU as: {_openvino_cpu_device_info(core)}")
    _out(f"  Logical cores       : {os.cpu_count()}")
    _out(f"  Torch threads used  : {torch.get_num_threads()}")
    mem = _total_memory_gb()
    _out(f"  Total memory (GB)   : {mem if mem is not None else 'unknown'}")
    _out(f"  GPU                 : N/A (CPU-only workload by design)")
    _out(f"  SW workload         : hardware_benchmark_flow.py (PyTorch CNN train, OpenVINO inference)")
    _out(f"  Training engine     : PyTorch {torch.__version__} (CPU)")
    _out(f"  Inference engine    : OpenVINO Runtime {ov.__version__} (CPU plugin)")
    _out(f"  OS / platform       : {platform.platform()}")
    # rapl = _rapl_package_files()
    # _out(f"  RAPL power readable : {'yes (' + str(len(rapl)) + ' package(s))' if rapl else 'NO -> power KPI unavailable, run as root'}")
    _out("=" * 78)


# ----------------------------------------------------------------------------------------
# 3. THE WORKLOAD  (a deliberately "heavy", CPU-bound CNN — train + inference)
# ----------------------------------------------------------------------------------------
# We use a small convolutional network on synthetic 3x32x32 ("CIFAR-shaped") data. Conv layers
# lean hard on Intel's oneDNN/AVX paths, so they make a good, scalable CPU stress test: a faster
# CPU will visibly win on both training time and inference throughput. Everything is seeded so
# the two machines do byte-identical work. Training runs in PyTorch (needs autograd); the
# trained weights are then converted to an OpenVINO IR and ALL inference measurements run
# through the OpenVINO CPU plugin.

class _BenchCNN(nn.Module):
    """A compact but compute-dense CNN. Sized to keep the CPU busy, not to win Kaggle."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                       # 32 -> 16
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                       # 16 -> 8
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                       # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def _make_synthetic_data(n_samples, num_classes, seed):
    """Deterministic synthetic image dataset. No download, identical on every machine."""
    _dbg(f"data: generating {n_samples} synthetic 3x32x32 samples (seed={seed}) ...")
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, 3, 32, 32), dtype=np.float32)
    y = rng.integers(0, num_classes, size=n_samples).astype(np.int64)
    # Bake a faint class-dependent signal in so training has something to chew on (keeps the
    # optimizer doing real work rather than thrashing on pure noise).
    for c in range(num_classes):
        x[y == c, c % 3] += 0.15 * c
    _dbg(f"data: done, tensor x={tuple(x.shape)} ~{x.nbytes / 1e6:.1f} MB")
    return torch.from_numpy(x), torch.from_numpy(y)


def _to_openvino(model, num_classes, core, run_idx):
    """Convert a trained (eval-mode) PyTorch model to an OpenVINO compiled model.

    Uses a dynamic batch dimension so the same compiled model serves both the large-batch
    throughput phase and the batch-size-1 latency phase without re-converting/re-compiling.
    """
    _dbg(f"run[{run_idx}]: OpenVINO: converting trained model to IR ...")
    example_input = torch.zeros(1, 3, 32, 32, dtype=torch.float32)
    t0 = time.perf_counter()
    ov_model = ov.convert_model(model, example_input=example_input)
    ov_model.reshape({0: [-1, 3, 32, 32]})   # dynamic batch dim
    compiled_model = core.compile_model(ov_model, "CPU")
    _dbg(f"run[{run_idx}]: OpenVINO: convert+compile took "
         f"{time.perf_counter() - t0:.2f}s")
    return compiled_model


def _single_run(run_idx, sizes, seed, core):
    """Execute ONE full benchmark iteration and return its KPI dict.

    Phases:
      (a) Training  — heavy load, measured as training throughput (samples/sec), in PyTorch.
          Also serves to warm caches/threads before we measure the inference KPIs.
      (b) OpenVINO conversion — trained weights -> IR -> compiled CPU model.
      (c) Inference throughput — large batches under sustained load -> KPI 1 (samples/sec),
          run through the OpenVINO compiled model.
          CPU power is sampled across THIS phase -> KPI 3 (watts).
      (d) Inference latency — many single-sample calls -> KPI 2 (per-request ms, p50/p95/mean),
          also run through the OpenVINO compiled model.
    """
    # Reproducible per run: same seed family on both machines, but each run varies a little
    # (run_idx) so we're not measuring the exact same cache-warmed state five times.
    _dbg(f"run[{run_idx}]: seeding torch & numpy with {seed + run_idx}")
    torch.manual_seed(seed + run_idx)
    np.random.seed(seed + run_idx)

    model = _BenchCNN(num_classes=sizes["num_classes"])
    n_params = sum(p.numel() for p in model.parameters())
    _dbg(f"run[{run_idx}]: built CNN with {n_params:,} params")
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # ---- (a) TRAINING  (PyTorch, needs autograd) -----------------------------------------
    x_tr, y_tr = _make_synthetic_data(sizes["train_samples"], sizes["num_classes"], seed + run_idx)
    bs = sizes["train_batch"]
    n_batches = sizes["train_samples"] // bs

    model.train()
    _dbg(f"run[{run_idx}]: TRAIN phase start — {sizes['epochs']} epoch(s) x {n_batches} batches "
         f"of {bs} (={sizes['epochs'] * n_batches * bs} samples) ...")
    # Heartbeat ~8 times per epoch regardless of batch count, so even a slow CPU shows liveness.
    hb = max(1, n_batches // 8)
    t0 = time.perf_counter()
    for ep in range(sizes["epochs"]):
        ep_t0 = time.perf_counter()
        last_loss = None
        for b in range(n_batches):
            xb = x_tr[b * bs:(b + 1) * bs]
            yb = y_tr[b * bs:(b + 1) * bs]
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            last_loss = loss.item()
            if b % hb == 0:
                _dbg(f"run[{run_idx}]:   epoch {ep + 1} batch {b}/{n_batches} loss={last_loss:.4f}")
        _dbg(f"run[{run_idx}]:   epoch {ep + 1}/{sizes['epochs']} done in "
             f"{time.perf_counter() - ep_t0:.2f}s (loss={last_loss:.4f})")
    train_secs = time.perf_counter() - t0
    train_samples_done = sizes["epochs"] * n_batches * bs
    train_throughput = train_samples_done / train_secs
    _dbg(f"run[{run_idx}]: TRAIN done in {train_secs:.2f}s -> {train_throughput:.1f} samples/s")

    # ---- (b) CONVERT TO OPENVINO ----------------------------------------------------------
    model.eval()
    compiled_model = _to_openvino(model, sizes["num_classes"], core, run_idx)
    output_port = compiled_model.output(0)

    # ---- (c) INFERENCE THROUGHPUT  (KPI 1) --------------------------------
    x_inf, _ = _make_synthetic_data(sizes["infer_samples"], sizes["num_classes"], seed + 999 + run_idx)
    x_inf_np = x_inf.numpy()
    ibs = sizes["infer_batch"]
    n_ibatches = sizes["infer_samples"] // ibs

    # meter = _PowerMeter()
    _dbg(f"run[{run_idx}]: THROUGHPUT phase start (OpenVINO) — {n_ibatches} batches of {ibs} ")
        #  f"(power metering {'ON' if meter.available else 'OFF'}) ...")
    t0 = time.perf_counter()
    # with meter:
    for b in range(n_ibatches):
        _ = compiled_model(x_inf_np[b * ibs:(b + 1) * ibs])[output_port]
    infer_secs = time.perf_counter() - t0
    infer_throughput = (n_ibatches * ibs) / infer_secs            # KPI 1: samples / sec
    # avg_power_w = meter.avg_watts                                 # KPI 3: watts (or None)
    _dbg(f"run[{run_idx}]: THROUGHPUT done in {infer_secs:.2f}s -> {infer_throughput:.1f} samples/s")

    # ---- (d) INFERENCE LATENCY  (KPI 2) ---------------------------------------------------
    # "Latency" = time to serve ONE request. We use batch-size-1 inference, which is the
    # standard way to express per-request responsiveness (distinct from bulk throughput).
    # Uses its own OpenVINO infer request so repeated single-sample calls don't pay any
    # python-object-creation overhead beyond the call itself.
    _dbg(f"run[{run_idx}]: LATENCY phase start (OpenVINO) — {sizes['latency_warmup']} warmup + "
         f"{sizes['latency_iters']} timed single-sample calls ...")
    single_np = x_inf_np[:1]
    infer_request = compiled_model.create_infer_request()
    for _ in range(sizes["latency_warmup"]):                 # warm up, don't time these
        infer_request.infer(single_np)
    per_call_ms = []
    for _ in range(sizes["latency_iters"]):
        t0 = time.perf_counter()
        infer_request.infer(single_np)
        per_call_ms.append((time.perf_counter() - t0) * 1000.0)

    per_call_ms.sort()
    latency_mean = statistics.fmean(per_call_ms)
    latency_p50 = per_call_ms[len(per_call_ms) // 2]
    latency_p95 = per_call_ms[min(len(per_call_ms) - 1, int(len(per_call_ms) * 0.95))]
    _dbg(f"run[{run_idx}]: LATENCY done -> mean={latency_mean:.3f} ms "
         f"p50={latency_p50:.3f} p95={latency_p95:.3f}")

    return {
        "train_throughput_sps": train_throughput,
        "infer_throughput_sps": infer_throughput,   # KPI 1
        "latency_mean_ms": latency_mean,            # KPI 2 (headline = mean)
        "latency_p50_ms": latency_p50,
        "latency_p95_ms": latency_p95,
        # "power_w": avg_power_w,                      # KPI 3 (None if RAPL unavailable)
        # "energy_j": meter.energy_joules,
    }


# ----------------------------------------------------------------------------------------
# 4. COMPOSITE SCORE + DELTA
# ----------------------------------------------------------------------------------------
def _composite_score(throughput_sps, latency_ms):
    """Fold the 3 KPIs into ONE 'higher-is-better' score (see module docstring).

        S = ( Throughput / Latency_ms ) ** (1/3)

    EDIT: NOT USING POWER
    """
    # if power_w and power_w > 0:
    #     return (throughput_sps / (latency_ms * power_w)) ** (1.0 / 3.0), 3
    return math.sqrt(throughput_sps / latency_ms), 2


# ----------------------------------------------------------------------------------------
# 5. THE FLOW FUNCTION
# ----------------------------------------------------------------------------------------
def func(
    num_runs=5,
    train_samples=6000,
    infer_samples=6000,
    epochs=1,
    train_batch=128,
    infer_batch=256,
    latency_iters=1000,
    latency_warmup=50,
    num_classes=10,
    seed=1234,
    baseline_score=None,
    num_threads=None,
    debug=True,
):
    """Run the full hardware benchmark and print/return its KPIs.

    Parameters (all have heavy-but-reasonable defaults; KEEP THEM IDENTICAL on both machines):
        num_runs        : how many times to repeat the whole workload, then average (default 5).
        train_samples   : synthetic training-set size per run (drives training load).
        infer_samples   : synthetic inference-set size (drives throughput KPI).
        epochs          : training passes per run (more = heavier).
        train_batch     : training mini-batch size.
        infer_batch     : batch size for the throughput phase (large = sustained load).
        latency_iters   : number of single-sample inferences timed for the latency KPI.
        latency_warmup  : untimed single-sample inferences before latency timing.
        num_classes     : output classes of the CNN.
        seed            : master RNG seed -> byte-identical data/work across machines.
        baseline_score  : the COMPOSITE SCORE S printed by the BASELINE machine's run. Pass it
                          on the NEW (Wildcat Lake) machine and this function prints the single
                          Delta % directly. Leave None on the baseline run.
        num_threads     : pin torch CPU threads (None = library default = all cores). Set the
                          SAME value on both machines if you want a thread-controlled comparison.
                          Note: this pins PyTorch's own thread pool (used for training); OpenVINO
                          manages its own CPU-plugin thread pool internally for inference.

    Returns a dict with the per-run results, the averaged KPIs, the composite score, and (if
    baseline_score was given) the headline Delta %.
    """
    print("[BENCH] >>> func() ENTERED — benchmark is running (this confirms the node fired)",
          flush=True)
    # Reset the global debug switch + wall-clock origin so [DBG +Xs] is relative to THIS call.
    global _DEBUG, _T0
    _DEBUG = bool(debug)
    _T0 = time.perf_counter()

    if num_threads:
        torch.set_num_threads(int(num_threads))
        _dbg(f"pinned torch CPU threads to {num_threads}")

    core = ov.Core()
    _dbg(f"OpenVINO Core created; available devices: {core.available_devices}")

    _dbg(f"func() start — num_runs={num_runs} train_samples={train_samples} "
         f"infer_samples={infer_samples} epochs={epochs} seed={seed}")
    print_system_info(core)
    _out(f"\nRunning {num_runs} iteration(s). Training: PyTorch. Inference: OpenVINO (CPU). "
         f"KPIs: throughput (up=good), latency ms (down=good)"
        #  f", power W (down=good).\n"
         )

    runs = []
    for i in range(num_runs):
        _dbg(f"========== RUN {i + 1}/{num_runs} starting ==========")
        run_t0 = time.perf_counter()
        m = _single_run(i+1, dict(
            train_samples=train_samples, infer_samples=infer_samples, epochs=epochs,
            train_batch=train_batch, infer_batch=infer_batch, latency_iters=latency_iters,
            latency_warmup=latency_warmup, num_classes=num_classes,
        ), seed, core)
        runs.append(m)
        _dbg(f"========== RUN {i + 1}/{num_runs} finished in "
             f"{time.perf_counter() - run_t0:.2f}s ==========")

        # power_str = f"{m['power_w']:.2f} W" if m["power_w"] is not None else "n/a (no RAPL)"
        _out(f"--- Run {i + 1}/{num_runs} ------------------------------------------------")
        _out(f"    Train throughput   : {m['train_throughput_sps']:>10.1f} samples/s  (PyTorch)")
        _out(f"    KPI1 Throughput    : {m['infer_throughput_sps']:>10.1f} samples/s  (OpenVINO inference)")
        _out(f"    KPI2 Latency       : {m['latency_mean_ms']:>10.3f} ms  (OpenVINO | mean | "
             f"p50 {m['latency_p50_ms']:.3f} | p95 {m['latency_p95_ms']:.3f})")
        # _out(f"    KPI3 Power         : {power_str:>14}"
            #  + (f"   ({m['energy_j']:.1f} J over inference phase)" if m['energy_j'] else ""))
        _out()

    # ---- AVERAGE THE KPIs ACROSS RUNS ---------------------------------------------------
    def avg(key):
        vals = [r[key] for r in runs if r[key] is not None]
        return statistics.fmean(vals) if vals else None

    avg_throughput = avg("infer_throughput_sps")
    avg_latency = avg("latency_mean_ms")
    # avg_power = avg("power_w")
    avg_train_tp = avg("train_throughput_sps")

    _dbg(f"averaging {num_runs} run(s): throughput={avg_throughput:.1f} "
         f"latency={avg_latency:.3f}")
    # "power={avg_power if avg_power is None else round(avg_power, 2)}")
    score, n_kpis = _composite_score(avg_throughput, avg_latency)
    _dbg(f"composite score S={score:.4f} using {n_kpis} KPI(s)")

    _out("=" * 78)
    _out(f"AVERAGE over {num_runs} run(s)   (these are the numbers to report to Intel)")
    _out("-" * 78)
    _out(f"  Train throughput      : {avg_train_tp:>10.1f} samples/s  (PyTorch)")
    _out(f"  KPI1 Throughput       : {avg_throughput:>10.1f} samples/s  (OpenVINO)")
    _out(f"  KPI2 Latency (mean)   : {avg_latency:>10.3f} ms  (OpenVINO)")
    # _out(f"  KPI3 Power            : {(f'{avg_power:.2f} W') if avg_power is not None else 'n/a (RAPL unreadable; run as root)':>10}")
    _out(f"  COMPOSITE SCORE S     : {score:>10.4f}   "
         f"({n_kpis}-KPI {'throughput/(latency*power)' if n_kpis == 3 else 'throughput/latency (NO power)'} geomean)")
    _out("=" * 78)

    result = {
        "runs": runs,
        "avg": {
            "train_throughput_sps": avg_train_tp,
            "infer_throughput_sps": avg_throughput,
            "latency_mean_ms": avg_latency,
            # "power_w": avg_power,
        },
        "composite_score": score,
        "composite_kpis": n_kpis,
        "delta_percent": None,
    }

    # ---- THE SINGLE DELTA %  (only when a baseline score is supplied) -------------------
    if baseline_score:
        delta = (score / baseline_score - 1.0) * 100.0
        result["delta_percent"] = delta
        verdict = "FASTER/BETTER" if delta >= 0 else "SLOWER/WORSE"
        _dbg(f"delta: this S={score:.4f} vs baseline S={baseline_score:.4f} -> {delta:+.2f}%")
        _out()
        _out("#" * 78)
        _out(f"#  DELTA vs baseline (baseline S = {baseline_score:.4f}, this machine S = {score:.4f})")
        _out(f"#  >>> SINGLE DELTA % = {delta:+.2f} %   ({verdict} overall)")
        _out("#" * 78)
    else:
        _dbg("no baseline_score supplied -> skipping delta computation")
        _out("\n(No baseline_score given — this looks like the BASELINE run. Note the")
        _out(" 'COMPOSITE SCORE S' above and pass it as baseline_score on the Wildcat Lake")
        _out(" machine to get the single Delta %.)")

    _dbg("func() complete")
    return result


# ----------------------------------------------------------------------------------------
# 6. ENTRY POINT  —  mlflow-engine trailing-assignment convention
# ----------------------------------------------------------------------------------------
# IMPORTANT: the engine does NOT run `if __name__ == "__main__"`. It execs this file in a
# namespace and reads back the variable named on the LEFT of the trailing `func(...)` call.
# So the line below is what actually runs the benchmark — both inside the engine AND when you
# run `python hardware_benchmark_flow.py` directly (module-level code runs in both cases).
#
# The lvalue `benchmark_result` is the node's OUT slot name — make sure the FunctionNode has
# an output slot called `benchmark_result` if you want to pass the result downstream (without
# one the engine just logs "(no outputs)", which is harmless — the KPIs still print).
#
# To get the Delta % on the Wildcat Lake machine, give it the baseline's COMPOSITE SCORE S.
# Either (a) set env var MLFLOW_BENCHMARK_BASELINE=<score> before the run, or (b) edit the
# call below to e.g. func(baseline_score=0.7421). On the baseline machine, leave it unset.
_baseline_env = os.environ.get("MLFLOW_BENCHMARK_BASELINE")
benchmark_result = func(
    baseline_score=float(_baseline_env) if _baseline_env else None,
    # baseline_score = 21.7505
)

# Compact machine-readable summary line (handy for grepping the logs / diffing two machines).
_out("JSON: " + json.dumps(benchmark_result["avg"] | {
    "composite_score": benchmark_result["composite_score"],
    "delta_percent": benchmark_result["delta_percent"],
}))