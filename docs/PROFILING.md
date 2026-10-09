# Profiling the GPU code, and where it diverges

This note explains how to profile the CUDA backend with NVIDIA Nsight Systems
and Nsight Compute, what to look at, where thread divergence comes from in
this problem, and how event-based tracking would reduce it.

The CUDA library is always compiled with `-lineinfo` (see `src/CMakeLists.txt`).
That flag maps machine instructions back to source lines in Nsight Compute and
costs nothing at run time.

## 1. Nsight Systems: the timeline

Nsight Systems answers *where does the wall time go?* It shows kernels, memory
copies and host activity on one timeline.

```bash
nsys profile -o optphot_records --force-overwrite true \
    ./build-gpu/apps/optphot --backend gpu --mode records --n_photons 1e7
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum optphot_records.nsys-rep
```

Open the `.nsys-rep` file in the Nsight Systems GUI to see the timeline.

What to look for:

* **Records mode is dominated by the device-to-host copy.** Each photon record
  is 29 bytes, so 10⁷ photons means 290 MB. The copy goes into pageable host
  memory, which the driver stages through a pinned buffer. Compare the
  `cudaMemcpy` bars with the kernel bars, or read
  `docs/figures/benchmark_*_gpu_breakdown.png`.
* **Nothing overlaps.** The chunks are processed strictly in sequence: kernel,
  copy, kernel, copy. Two standard improvements, not implemented in v1:
  - **pinned host buffers** (`cudaMallocHost`), which make the copy faster;
  - **two CUDA streams**, so that chunk k+1 is transported while chunk k is
    copied back.
* **Tally mode has no bulk transfer.** Its timeline is one long kernel plus a
  few-kB copy. This mode is the fair measure of transport throughput.
* **Context creation.** The first CUDA call of a process takes 0.1–1 s. The
  backend calls `cudaFree(nullptr)` before starting its timers, and the
  benchmark does a warm-up run.

## 2. Nsight Compute: inside the kernel

Nsight Compute answers *how efficiently does the kernel use the GPU?*
Profile a single launch:

```bash
ncu --kernel-name regex:transport_tally --launch-count 1 \
    --section SpeedOfLight --section Occupancy --section LaunchStats \
    --section WarpStateStats --section SourceCounters \
    ./build-gpu/apps/optphot --backend gpu --mode tally --n_photons 1e7
```

To measure divergence directly, collect:

```bash
ncu --kernel-name regex:transport_records --launch-count 1 \
    --metrics smsp__thread_inst_executed_per_inst_executed.ratio,smsp__sass_average_branch_targets_threads_uniform.pct \
    ./build-gpu/apps/optphot --backend gpu --n_photons 1e6
```

`smsp__thread_inst_executed_per_inst_executed.ratio` is the average number
of active threads per executed warp instruction. 32 means no divergence at
all. Compare it with the estimate in section 3.

What to look for:

* **LaunchStats / Occupancy.** The whole photon history is inlined into one
  kernel, so register use per thread is fairly high, and that limits how
  many warps fit on an SM.
  - Check the "Registers Per Thread" value and the occupancy limiter.
  - `__launch_bounds__` or `-maxrregcount` can trade registers for occupancy.
    Measure before believing that it helps.
* **SpeedOfLight.** This kernel is expected to be **compute / latency bound**,
  not memory bound.
  - There is almost no memory traffic apart from the final record stores.
  - The Philox rounds are integer multiplies.
  - The physics is FP32 arithmetic plus a few transcendental functions.
* **WarpStateStats.** A large share of stalls on "selected / not selected"
  together with low active threads per warp means lanes are idle because of
  divergence, not because of memory latency.
* **SourceCounters.** Divergent branches are attributed to source lines
  (thanks to `-lineinfo`). Expect the top entries to be the step loop
  condition and the boundary-type branches in `transport.hpp`.

## 3. Where thread divergence occurs in this problem

A warp executes 32 threads in lockstep. When those threads need different
code, or different numbers of loop iterations, the hardware serialises the
paths and masks off the lanes that are not on the current path.

1. **History length (the dominant effect).** Thread *i* owns photon *i* until
   that photon dies, so a warp runs until its longest history ends.
   - Most photons end after one or two steps: they are detected, absorbed at
     a black wall, or refracted out of a polished face.
   - A photon trapped by total internal reflection can bounce hundreds of
     times while 31 lanes sit idle.

   `python/divergence_estimate.py` measures this effect from photon records,
   so it runs without a GPU. It computes Σ steps / Σ_warps (32 × max steps in
   the warp):

   | setup (10⁶ photons) | mean steps | longest history | history-based warp efficiency |
   |---|---|---|---|
   | black walls | 1.00 | 1 | 100% |
   | polished cube (default) | 2.34 | 284 | **16%** |
   | Lambertian wrap, volume source | 4.64 | 64 | **27%** |

   So in the default polished cube only about one lane in six does useful
   work. The cause is the long tail of TIR-trapped photons (see
   `docs/figures/divergence_history_length.png`). This estimate assumes equal
   cost per step; Nsight Compute's active-threads metric gives the measured
   value.

2. **Different interaction types in the same step.** In one loop iteration
   some lanes absorb, some scatter (a Rayleigh rotation with `cbrt` and
   `sincos`), and most reach a boundary. At the boundary, some lanes compute
   Fresnel coefficients for the readout face and others for a side face; some
   reflect, some transmit, and some sample a Lambertian direction. Each branch
   is short, so this costs much less than effect 1.

3. **Avoided by design:**
   - Rayleigh and Lambertian sampling use closed-form inverse CDFs, with no
     rejection loops of random trip count.
   - Face navigation is branch-light: three `if`s with arithmetic, and no
     search.
   - The surface model is a kernel parameter, so every thread takes the same
     side of that `if`.

## 4. How event-based tracking reduces it, and what it costs

In an **event-based** (or "stepping-loop-on-the-host") design, the photon
state lives in global memory and each kernel launch advances *all alive
photons by one step*.

1. A `step` kernel samples the distances to absorption, scattering and the
   boundary for every alive photon. It moves each photon and records which
   event comes next.
2. The photons are **partitioned or sorted by event type** (a stream
   compaction or a radix sort on the event code). Then dedicated kernels
   (`absorb`, `scatter`, `boundary_readout`, `boundary_side`) each run over a
   contiguous, homogeneous set of photons, so every warp executes the same
   code.
3. Dead photons are **compacted away**, so warps stay full even while the
   long TIR-trapped tail is being transported. That tail is exactly the
   effect that limits the history-based kernel to 16–27% here.
4. New photons can be **refilled** into the freed slots from a queue. This is
   the standard way to keep the GPU saturated when photons are generated by
   upstream physics (Geant4 scintillation/Cerenkov "gensteps", as in Opticks).

The costs:
- Every step now reads and writes the photon state through global memory.
  This is where a **structure-of-arrays** layout becomes essential (coalesced
  per-field access), whereas the history-based kernel keeps the state in
  registers.
- There is kernel-launch and compaction overhead on every step.
- The code is more complex.

How much does compaction buy back? `divergence_estimate.py` replays the
recorded history lengths, compacting the alive photons into full warps every
*k* steps. This assumes equal cost per step and ignores the cost of
compaction itself.

| setup | history-based | compact every 8 steps | every 4 | every 2 | every step |
|---|---|---|---|---|---|
| polished cube (default) | 16% | 30% | 49% | 74% | ~100% |
| Lambertian wrap | 27% | 50% | 72% | 89% | ~100% |

The tail is heavy (TIR-trapped photons), so **compacting once is not
enough**. A single compaction after the first 2–5 steps only reaches about
27–30% in the polished cube, because the survivors' lengths are still very
uneven. The efficiency comes from compacting *repeatedly*. A practical
design is therefore:
- a fused kernel that runs a few steps per launch (amortising launch and
  state traffic);
- followed by compaction of the survivors;
- repeated until no photons are left, with new photons refilling the freed
  slots.

Choosing *k* is a trade-off between lane utilisation and the cost of
moving photon state through global memory. Only measurement settles it.

**Measured** (GPU v2, [`src/gpu/gpu_event.cu`](../src/gpu/gpu_event.cu); full
table and discussion in the README, §7). This version uses a fused kernel
that runs up to *k* steps per launch, with warp-aggregated compaction and no
refill.
- The lane efficiencies above are reproduced exactly by counters in the
  kernel.
- The speed-up is much smaller than the efficiency gain: at best 1.10×
  (polished, k = 8) and 1.27× (Lambertian, k = 4), and k = 1 is slower than
  history-based.
- Most launches in the tail have too few photons to fill the GPU, and every
  launch costs a host round trip of about 50 µs.
- Refill and a device-side loop are what would turn the lane efficiency into
  speed.
