# GPU concepts used in this project — learning notes

These notes are for someone who knows C++ and Monte Carlo transport but is
new to GPUs (which is how this project started). Each section explains one
concept in plain language and points to where it appears in the code.

The single most useful mental model: **a GPU is a throughput machine.** It
hides the latency of each individual operation by keeping tens of thousands
of threads in flight, and it runs those threads in groups of 32 that execute
the same instruction at the same time. Almost every design decision below
follows from those two facts.

---

## 1. Host, device, and the same code compiled twice

The CPU is the *host* and the GPU is the *device*. They have separate memory.
Device code is compiled by `nvcc` for the GPU's instruction set.

A function marked `__host__ __device__` is compiled **twice**: once for the
CPU and once for the GPU. This project uses that to keep a single copy of the
physics:

* `include/optphot/hd.hpp` defines `OPT_HD`. Under `nvcc` it expands to
  `__host__ __device__`; in a plain C++ compiler it expands to nothing, so the
  CPU-only build needs no CUDA at all.
* All of the physics (`transport.hpp`, `fresnel.hpp`, `sampling.hpp`,
  `geometry.hpp`, `philox.hpp`) is header-only and marked `OPT_HD`. The CPU
  backend (`src/cpu/cpu_backend.cpp`) and the CUDA kernels
  (`src/gpu/gpu_backend.cu`) both call the very same `transport_photon()`.
* Where host and device need *different* implementations, the code checks
  `__CUDA_ARCH__`, which is defined only during the device compilation pass.
  Examples: `mulhilo` in `philox.hpp` uses the `__umulhi` intrinsic on the
  device and a 64-bit multiply on the host; `float_bits` in `pmath.hpp` does
  the same with `__float_as_uint` and `memcpy`.

Restrictions to know about in device code:
- no exceptions, no `std::vector`, no virtual calls across the host/device
  boundary;
- only device-callable math functions (`sqrtf`, `logf`, … are fine).

## 2. Kernels, threads, blocks, grids

A **kernel** is a `__global__` function, launched from the host with
`kernel<<<grid, block>>>(args)`. The launch creates `grid × block` threads,
and every thread runs the whole kernel with its own indices:

```cpp
const uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;   // global thread id
if (i >= n) return;                                          // last block may be partial
```

* **Block:** up to 1024 threads that run on the same streaming
  multiprocessor (SM). They can share fast *shared memory* and synchronise
  with `__syncthreads()`. Typical sizes are 128–256. This project uses 256
  (`block_size`, configurable).
* **Grid:** all the blocks of one launch. Blocks are independent and can run
  in any order. That independence is what lets the same code scale from a
  laptop GPU to an H100.

Where to look:
* `transport_records_kernel`: one thread per photon. The grid is
  `ceil(n / 256)` blocks, and thread *i* transports photon *i*.
* `transport_tally_kernel`: a **grid-stride loop**
  (`for (i = tid; i < n; i += gridDim.x * blockDim.x)`). The grid is sized to
  what fits on the GPU at once, from `cudaOccupancyMaxActiveBlocksPerMultiprocessor`,
  and each thread processes many photons. This amortises the per-block setup
  (initialising and flushing the shared-memory histograms).

## 3. Warps and SIMT

The hardware groups each block into **warps** of 32 consecutive threads. A
warp issues one instruction at a time for all 32 lanes (SIMT: single
instruction, multiple threads).

A GPU has many SMs, and each SM keeps dozens of warps resident. Whenever one
warp waits for memory, the SM switches to another warp at no cost. That is
how latency is hidden, and why you want *many* threads: for example, 10⁶
photons rather than 10³. The benchmark shows exactly this: GPU throughput
keeps rising with N until the device is full.

**Occupancy** is the fraction of the SM's warp slots actually in use. It is
limited by registers per thread, shared memory per block and block size. Our
kernel inlines a whole photon history, so it uses many registers. Nsight
Compute reports the limiting factor (see `docs/PROFILING.md`).

## 4. Thread divergence

If lanes of a warp take different branches, the warp runs **both** paths one
after the other, masking off the lanes that are not on the current path.
Loops behave the same way: the warp keeps iterating until its *last* lane
is done.

In this problem the dominant effect is **history length**:
- most photons finish after 1–2 steps;
- a photon trapped by total internal reflection may take hundreds of steps,
  and the other 31 lanes of its warp wait.

`python/divergence_estimate.py` measures this from photon records (it needs
no GPU). Only about 16% of lanes do useful work in the default polished
cube, and about 27% with a Lambertian wrapping.

Choices made to limit divergence:
* **Closed-form sampling instead of rejection loops.** Rayleigh `cos θ` comes
  from the exact Cardano inversion (`sample_rayleigh_cos`), and the
  Lambertian direction from `cos θ = sqrt(u)`. A rejection loop would give
  each lane a random number of iterations.
* **Branch-light geometry.** `distance_to_box_exit` is three small `if`s and
  arithmetic.
* **Uniform branches are free.** `p.surface` is the same for every thread
  (it is a kernel parameter), so the `if (p.surface == …)` chain never
  diverges.

The remedy for history-length divergence is **event-based tracking**: compact
the alive photons and sort them by next interaction. `docs/PROFILING.md`
discusses it and quantifies the gain.

## 5. The memory hierarchy

Fastest to slowest:

| memory | scope | used here for |
|---|---|---|
| registers | one thread | the whole photon state (position, direction, time, RNG buffer) during a history |
| local memory | one thread, but physically in DRAM | avoided, see below |
| shared memory | one block, on-chip | block-private tally histograms (`s_hist` in `transport_tally_kernel`) |
| constant bank | read-only, broadcast | kernel arguments, i.e. `SimParams` passed by value |
| global memory | whole device (DRAM) | output record buffers, global tallies |

Two details that matter in this code:

* **Avoiding local memory.** An array indexed with a runtime value usually
  cannot live in registers, so the compiler spills it to *local memory*,
  which is slow DRAM. `PhiloxStream` buffers four random numbers, but keeps
  them in four named members and *shifts* them instead of indexing an array
  with a counter (`philox.hpp`). For the same reason, `face_normal` and
  `snap_to_face` compute components with ternaries rather than `v[axis]`.
* **Parameters via the constant bank.** `SimParams` is a small, trivially
  copyable struct passed by value to the kernel. Kernel arguments live in a
  constant bank, which is cached and *broadcast* when all lanes read the
  same address. That is exactly our access pattern, since every thread reads
  `p.n_scint`.

## 6. Coalescing and structure-of-arrays

Global memory is accessed in 32-byte sectors. When the 32 lanes of a warp
access **consecutive addresses**, the hardware *coalesces* them into a few
wide transactions. When they access scattered addresses, each lane can cost
its own transaction.

Thread *i* writes photon *i*'s record:
* With an **array of structs** (`PhotonRecord recs[N]`), lane *i* writes
  `recs[i].t` at address `base + i·29`. Those addresses are strided, so the
  writes are poorly coalesced.
* With a **structure of arrays**, one array per field, lane *i* writes `t[i]`
  at `base + 4·i`. The warp writes 128 contiguous bytes, which is perfectly
  coalesced.

So `PhotonRecords` (`backend.hpp`) and `DeviceRecords` (`gpu_backend.cu`) are
SoA. The same layout is also exactly what NumPy wants: each field becomes one
`.npy` file.

In this history-based kernel the photon *state* stays in registers, so SoA
only matters for the output. In an event-based design the state lives in
global memory between kernels, and SoA becomes essential for every field.

## 7. Atomics and reductions

When many threads add to the same counter, a plain `+=` loses updates (a race
condition). `atomicAdd` makes the read-modify-write indivisible, but
contended atomics on one address serialise.

The tally kernel uses the standard **privatisation** pattern:
1. Each block keeps its own histograms in shared memory (`s_hist`).
   Shared-memory atomics are on-chip and only contend within the block.
2. After the photon loop, each block adds its non-zero bins to the global
   histograms once (one global `atomicAdd` per bin per block).

Two design choices worth defending in an interview:
* **Integer counters.** Floating-point addition is not associative, so float
  atomics give results that depend on the (non-deterministic) order in which
  threads arrive. All tallies here are integers, so the GPU tally is
  bit-for-bit reproducible. A test checks this with two different block
  sizes.
* **Records vs atomics.** Records mode needs no atomics at all: each thread
  owns its output slot. The price is memory and a large device-to-host copy.
  Tally mode is the production-style choice: kilobytes leave the GPU instead
  of gigabytes.

A possible next step is *warp-aggregated* atomics. The 32 lanes first
combine their updates with warp shuffles (`__shfl_down_sync`,
`__match_any_sync`), so only one lane per distinct bin issues the atomic.

## 8. Random numbers on a GPU: counter-based streams

A conventional generator (Mersenne Twister, cuRAND's XORWOW) has *state* that
must be stored, seeded and advanced. Giving each of 10⁸ photons its own
independent state is awkward, and results then depend on which thread
handled which photon.

A **counter-based** generator such as Philox (`philox.hpp`) is a pure
function: `(counter, key) → 128 random bits`. It is a few rounds of integer
multiplies and XORs, which is cheap on GPUs.
- Photon *i* uses counter `{block, 0, i_lo, i_hi}` and key = the run seed,
  so it has its own stream that is independent of every other photon's.
- That stream is identical whichever thread, GPU or CPU core processes the
  photon, and in whatever order.

This is what makes the photon-by-photon CPU/GPU comparison possible at all.
The tests check the Random123 known-answer vectors and the stream properties
(`tests/test_rng.cpp`).

Floats are made from the top 24 bits, as `k·2⁻²⁴` in (0, 1]. That value is
exactly representable, never zero (safe for `-log u`), and identical on both
sides.

## 9. Host–device memory transfers

`cudaMalloc` allocates device memory and `cudaMemcpy` copies across the PCIe
(or NVLink) bus. That bus is much slower than either side's own memory: about
10–25 GB/s, against hundreds of GB/s for device memory.

* **Design for no transfer.** Photons are *generated on the device* from
  `(seed, photon id)`, and the parameters travel as kernel arguments. So
  there is essentially no host-to-device traffic (`h2d_ms` ≈ 0 in the
  benchmarks).
  - In a real Geant4 integration the host would upload the "gensteps" (the
    scintillation and Cherenkov steps that produce photons, as in Opticks).
* **Device-to-host dominates records mode.** The copy is 29 bytes per
  photon, see `benchmark_*_gpu_breakdown.png`.
* **Chunking** (`run_gpu_records`, `chunk` parameter) bounds device memory
  for large N.
* **Pageable vs pinned memory.** `std::vector` is *pageable* host memory, so
  the driver copies through an internal pinned buffer. Allocating with
  `cudaMallocHost` (*pinned*) is faster and enables `cudaMemcpyAsync`.
  Together with **streams**, the copy of chunk *k* could then overlap with
  the transport of chunk *k+1*. This is listed as a next step.

## 10. Timing GPU code correctly

Kernel launches are **asynchronous**: the host continues immediately. Timing
a launch with `std::chrono` measures only the launch, unless you
synchronise.

* `EventPair` in `gpu_backend.cu` records `cudaEvent`s before and after the
  kernel, and before and after the copies. `cudaEventElapsedTime` gives the
  GPU-side duration, and `cudaEventSynchronize` waits for it.
* The **first CUDA call** in a process creates the context, which takes
  0.1–1 s. `ensure_context()` does it before the timers start, and
  `optphot_bench` also runs a warm-up launch.

## 11. Floating point: why CPU and GPU histories can differ

Both sides use IEEE-754 single precision, but bit-identical results are
**not** automatic:

* **FMA contraction.** `nvcc` fuses `a*b + c` into a single fused
  multiply-add by default (`--fmad=true`). That rounds once instead of
  twice, so it is more accurate but *different*. On the host we pass
  `-ffp-contract=off`, so x86 builds never fuse. The option
  `-DOPTPHOT_CUDA_FMAD=OFF` passes `--fmad=false` to `nvcc`.
* **Math library.** `logf`, `sinf`, `cosf` and `cbrtf` are accurate to 1–2
  ulp, but glibc, MSVC/MinGW and CUDA's libdevice are different
  implementations, so their last bits differ for some inputs. In contrast,
  `+ − × ÷ sqrt` are correctly rounded everywhere, provided `nvcc` keeps its
  defaults `-prec-div=true` and `-prec-sqrt=true`.
* **Fast math** (`--use_fast_math` / `-ffast-math`) would additionally:
  - flush denormals to zero;
  - use approximate division, square root and transcendentals;
  - and, on the host, allow reassociation.

  It is deliberately **not** used.

The consequence:
- A last-bit difference in a position is harmless.
- But sooner or later a comparison such as `u <= R` or "which face is
  nearer" flips, and from then on the two histories are different, equally
  valid Monte Carlo histories.

`OPTPHOT_PORTABLE_MATH` (`pmath.hpp`) replaces the four transcendental
functions with versions built only from correctly rounded operations. With
FMA contraction also disabled, every operation is then exactly specified by
IEEE-754, and the CPU and GPU **do** agree to the last bit. On a GTX 1660 Ti,
all 2·10⁶ records of a Lambertian + Rayleigh setup were identical, including
against a CPU build from a different compiler (GCC vs MSVC). The GPU test
asserts this in that configuration. The price was about 8–10% of GPU kernel
time.

Measured with the default build (FMA on, platform math), CPU vs GPU with the
same seed: 3 of 8·10⁶ histories took a different discrete path, but only
12–40% of records were bitwise identical. The same effect is visible on the
CPU alone (library math vs portable math, same seed, 10⁶ photons): 100% of
histories take identical discrete paths, but only 23% of
records are bitwise identical.

**Single vs double precision.** Consumer and cloud GPUs (GTX/RTX, T4) run
FP64 at 1/32 or 1/64 of the FP32 rate. Everything here is `float`, which is
ample for millimetre geometry: the float epsilon times 50 mm is about 3 nm.

## 12. Error handling

CUDA API calls return a `cudaError_t`, which `CUDA_CHECK` turns into an
exception with file and line. Kernel launches return nothing, and errors
arrive in two ways:
* **Launch errors** (bad configuration, too much shared memory) are
  reported by `cudaGetLastError()` right after the launch.
* **Execution errors** (an illegal address inside the kernel) appear at the
  next synchronising call, here `cudaEventSynchronize`. `compute-sanitizer`
  (`compute-sanitizer ./build-gpu/apps/optphot --backend gpu --n 1e5`) finds
  their cause.

## 13. Building CUDA with CMake

* `enable_language(CUDA)` plus `.cu` sources. CMake drives `nvcc` and passes
  the host compiler.
* `CMAKE_CUDA_ARCHITECTURES` selects the instruction sets to compile for:
  - `75` is Turing (T4, GTX 16xx);
  - `80` is A100, `86` is RTX 30xx, `89` is L4 / RTX 40xx.

  Code built for an older architecture also runs on newer GPUs through the
  embedded PTX, which the driver JIT-compiles.
* `-lineinfo` keeps source-line information for Nsight Compute without
  slowing the code.
* The GPU code is a separate static library (`optphot_gpu`). C++ test and
  application files (compiled by the host compiler) call it through the
  plain host API in `backend.hpp`. Catch2 never goes through `nvcc`.
