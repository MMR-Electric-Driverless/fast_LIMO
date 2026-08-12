# CPU optimizations — what changed, why, and what it means for CUDA

Reference for commit `cc694f0` (*cpu optimizations*) on branch `cuda-humble`.

This document has two jobs:

1. explain each change well enough that someone can re-derive it, re-measure it, or
   revert it on purpose;
2. record which of those changes exist *specifically* so that a CUDA backend can be
   dropped in later without rewriting the localizer.

Most of the reasoning is also inline in the code — this is the map, not a replacement
for it.

---

## 0. TL;DR

Measured on-vehicle, 20 Hz Hesai, same route, same bag:

| | octree (before) | hashgrid (after) | speedup |
|---|---|---|---|
| scan (end-to-end) | 21.39 ms | 13.88 ms | 1.54× |
| match (kNN + plane fit) | 10.68 ms | 4.68 ms | 2.28× |
| map insert | 4.09 ms | 0.26 ms | 15.7× |
| CPU cores used | 1.30 | 0.83 | — |
| map RAM | 67.4 MB | 53.2 MB | — |

The changes fall into four groups:

- **Instrumentation** — a per-stage timing breakdown, without which none of the rest
  was decidable.
- **The map structure** — incremental octree replaced by an iVox-style spatial hash,
  behind an interface so both can be A/B'd.
- **Allocation removal on the hot path** — kNN results, plane fitting, and the sweep
  sort.
- **Threading** — where the process actually spends its cores.

---

## 1. Instrumentation first

`Localizer.hpp` gained two small types:

- [`StageTimes`](../include/fast_limo/Modules/Localizer.hpp) — a per-scan wall-clock
  breakdown in milliseconds, one field per pipeline stage, plus `iterations` (KF inner
  passes actually executed) and `pc2match` (points handed to the matcher).
- [`ScopedTimer`](../include/fast_limo/Modules/Localizer.hpp) — RAII; **accumulates**
  into its target rather than assigning, so a stage entered once per KF iteration sums
  correctly over the scan.

`stages` is public on `Localizer` precisely so that `IKFoM::h_share_model()` — which
runs on the same thread, inside the iterated KF update — can charge the kNN half and
the Jacobian half separately. It is written only from the LiDAR callback, so it needs
no synchronization.

The breakdown is printed by `debugVerbose()` on the existing performance board.
`match` and `jacobian` are subsets of `kf`, summed over iterations.

Two related fixes shipped with it:

- **The debug thread is gone.** `debugVerbose()` used to be launched on a detached
  `std::thread` *per scan*. At 20 Hz that is 20 thread creations per second, churning
  the scheduler to produce output no human reads at that rate — and the detached thread
  raced against the LiDAR thread on the (non-thread-safe) stat buffers it reads. It now
  runs inline, rate-limited to ~1 Hz off `scan_stamp` (with a reset if the bag loops).
- **The board reports `pc2match` against `MAX_NUM_PC2MATCH`**, which is the fastest way
  to tell whether the matcher is being throttled by that cap.

> **Why this matters for CUDA:** the stage split is the acceptance test for a GPU
> backend. A device-resident map must be compared stage-by-stage against these numbers,
> not against end-to-end scan time, or a win in `match` gets hidden by an unchanged
> `deskew` and a new transfer cost that nobody attributed.

---

## 2. The map: octree → hash grid

### 2.1 Why the octree lost

Two operations dominate a scan: *insert this scan* and *give me the k nearest map
points*. Together ~60% of a scan's compute. Both are hostile to a tree:

- **insert** recursively subdivides and heap-allocates child arrays;
- **kNN** descends pointers to a data-dependent depth, so every level is a cache miss.

A synthetic single-threaded benchmark had originally concluded the *opposite* — that
the octree was faster. That benchmark batch-built the tree once via `initialize()`
instead of inserting one scan per frame via `update()`. Batch build is the octree's
best case and is not what this pipeline does. The incremental subdivision, plus the
cache pressure of six threads walking a large tree, is where the tree actually loses.

**Keep this in mind when benchmarking the CUDA backend: reproduce the incremental,
multi-threaded access pattern, or the measurement will lie in the same way.**

### 2.2 What replaced it

[`HashGrid.hpp`](../include/fast_limo/Objects/HashGrid.hpp) — an open-addressed spatial
hash of fixed-capacity voxels (iVox-style).

```
key (kx,ky,kz) = floor(p / voxel_size)          // floor, not truncation:
hash           = (kx*73856093) ^ (ky*19349669)  // truncation folds -0.4 and +0.4
                 ^ (kz*83492791)                // into one cell and mirrors the
idx            = hash & mask_                   // grid about the origin
```

Linear probing on collision, re-checking `(kx,ky,kz)` on each step. Load factor kept
under 0.7 by `reserveFor()`, called *before* the insert loop, so the probe always
terminates.

Insertion is a hash probe plus an array write. kNN scans a bounded set of neighbouring
cells whose points sit contiguously in one flat vector. No pointers, no recursion, no
allocation on either path.

### 2.3 The slot / pool split — the part worth understanding

A first version inlined `VOXEL_CAPACITY` `Vector3f`s directly into the slot. That made
each slot 260 B, and the (necessarily over-provisioned) table hundreds of MB. Every
probe missed cache and **both insert and query lost to the octree**.

The shipped layout separates them:

```
Slot  = { int32 kx, ky, kz; uint32 count; uint32 offset; }   // 20 bytes
table_ : vector<Slot>            // over-provisioned, power-of-two, probed
used_  : vector<uint8_t>         // occupancy flags
pool_  : vector<Vector3f>        // VOXEL_CAPACITY entries per *occupied* voxel
cursor_: vector<uint32_t>        // ring write position, one per occupied voxel
```

Consequences, all of them deliberate:

- Only **occupied** voxels cost point storage; the table stays cache-resident.
- `pool_` is **never reallocated by a rehash**, so `Slot::offset` survives table growth.
  A rehash moves 20-byte slots only.
- `VOXEL_CAPACITY = 20` is compile-time fixed. This bounds map memory, keeps every voxel
  a flat POD, and **doubles as the map's downsampling policy**. The runtime
  `max_points_per_voxel` may be lower, never higher.
- A full voxel overwrites its oldest point via `cursor_` (ring). The map tracks the
  recent scene instead of freezing on whatever was seen first.

### 2.4 kNN and cell pruning

`knn()` walks a fixed neighbour-offset table and, before touching a cell:

```cpp
if (heap.full() && cellMinDistSq(query, cx, cy, cz) >= heap.worstDist())
    continue;   // nothing in that box can beat the current k-th best
```

This is what stops a 27-cell scan from costing 27 cells of distance computations. It
only works because `NEIGHBORS_27` is **ordered by distance from the centre** — centre,
6 faces, 12 edges, 8 corners — so the heap is already tight by the time the far cells
come up for their box test. Do not reorder that table.

### 2.5 Tuning: the recall table

The shipped settings are measured, not defaults. Map points sit ~`leafSize` (0.5 m)
apart, so recall depends on the ratio `voxel_size / spacing`:

| voxel_size | neighbors | recall |
|---|---|---|
| 0.5 m | 7 | 0.45 — only 24% of queries return 5 points |
| 1.0 m | 7 | 0.72 |
| **1.5 m** | **27** | **1.00** ← shipped |

7 cells (face-adjacent only) **never** reaches full recall at any voxel size: it omits
the diagonal neighbours, where a fair share of the true 5-NN live.

### 2.6 The backend seam

[`MapBackend.hpp`](../include/fast_limo/Objects/MapBackend.hpp) defines `IMapBackend`
with four operations: `configure`, `size`, `add`, `knn`. Two implementations —
`OctreeMapBackend` (kept as the reference peer) and `HashGridMapBackend`. Selection is
one string in `params.yaml` (`iKFoM.Mapping.backend`), resolved by `makeMapBackend()`;
an unknown name falls back to the octree with a warning rather than failing, so a stale
config cannot take the vehicle down.

`LimoWrapper.cpp` reads the new parameters **defensively** (`has_parameter`), so a
config file predating this commit still launches on the octree exactly as before.

**This interface is where the CUDA map lands.** See §6.

---

## 3. Allocation removal on the hot path

The kNN + plane-fit path runs roughly **30k times per scan, per KF iteration, across
every OpenMP thread**. A single heap allocation there costs more than the search
itself, and worse, it serializes the threads on the malloc arena.

### 3.1 Shared, stack-resident kNN results

[`MapTypes.hpp`](../include/fast_limo/Objects/MapTypes.hpp):

- `MAX_KNN = 16` — a hard cap on `k` for any backend. Every shipped config uses
  `NUM_MATCH_POINTS == 5`, so a small fixed cap lets both the search collector and its
  results live on the stack.
- `KnnResult` — fixed-capacity `points[]` / `dists[]` (**squared** distances, sorted
  ascending) + `count`.
- `KnnHeap` — bounded collector. Insertion sort, not a real heap: at k=5 the linear
  shift beats a branchy sift-down and keeps the output sorted for free.

`Mapper::set_config()` clamps `NUM_MATCH_POINTS` to `MAX_KNN` **loudly** — silently
exceeding it would starve every plane fit, since `enough_points()` could never be
satisfied.

Critically, the octree was *also* converted to use `KnnHeap`/`KnnResult` (its private
`Heap`/`DistancePoint` types were deleted). Both backends therefore return identically
ordered results, which is the only thing that makes an octree-vs-hashgrid A/B — or
later, a CPU-vs-GPU A/B — a comparison of structures rather than of two different
notions of "the k nearest".

### 3.2 Plane fitting without the heap

[`Plane.cpp`](../include/fast_limo/Objects/Plane.cpp) before: build a dynamically-sized
`A (n×3)` and `b (n×1)`, then `A.colPivHouseholderQr().solve(b)`. Four separate heap
allocations per call, per point, per iteration.

After: accumulate the normal equations in place.

```cpp
Eigen::Matrix3f AtA = Eigen::Matrix3f::Zero();
Eigen::Vector3f Atb = Eigen::Vector3f::Zero();
for (int j = 0; j < n; j++) {
    const Eigen::Vector3f& a = pts[j];
    AtA.noalias() += a * a.transpose();
    Atb -= a;                       // b(j) == -1
}
Eigen::Vector3f normvec = AtA.ldlt().solve(Atb);
```

Both are least-squares solutions of the same system (`x·p + 1 = 0`), but `AtA` is a
fixed 3×3 that stays in registers.

**One correctness fix rides along and must not be dropped in any port.** Degenerate
neighbourhoods — collinear or coincident map points — leave `AtA` singular, and the
solve then yields inf/NaN. `estimate_plane()` now returns `false` on a non-finite or
near-zero norm. This is mandatory: `plane_eval()` compares with `>`, which is **false
for NaN**, so a NaN normal would otherwise be accepted as a *perfect* plane.

Also removed: the `centroid` member and `get_centroid()`, which nothing consumed.

`Plane`'s constructor now takes raw pointers (`const Eigen::Vector3f*`, `const float*`,
`int n`) instead of `MapPoints` + `std::vector<float>`, so the caller can pass the stack
buffers inside `KnnResult` directly. `close_enough()` relies on `sq_dists` being sorted
ascending — the last entry is the farthest.

### 3.3 The sweep sort that produced nothing

`deskewPointCloud()` used to `std::partial_sort_copy` all ~20k points by timestamp,
through a `std::function` comparator selected per sensor type.

Nothing downstream consumes that order:

- the deskew loop looks every point up **independently** via
  `binary_search_tailored()` over `frames` (the IMU state list — not the points);
- the voxel grid reorders the cloud anyway.

The sort's only product was its last element: the maximum extracted time. So it became
a `std::copy` plus an OpenMP `reduction(max:)`:

```cpp
#pragma omp parallel for reduction(max:max_point_time) num_threads(this->num_threads_)
for (size_t k = 0; k < deskewed_scan_->points.size(); k++) {
    double t_k = extract_point_time(deskewed_scan_->points[k]);
    if (t_k > max_point_time) max_point_time = t_k;
}
```

O(n) instead of O(n log n) `std::function` comparisons. The four `point_time_cmp`
lambdas were deleted; only `extract_point_time` remains, and it is a **pure function of
a point's own time field** — it never looks at neighbours or indices. That property is
what makes the sweep order irrelevant, and it is also what makes deskew trivially
parallel on a GPU.

---

## 4. Threading

### 4.1 Dynamic scheduling in `Mapper::match()`

```cpp
#pragma omp parallel for num_threads(this->num_threads_) schedule(dynamic, 64)
```

kNN cost per point varies by an order of magnitude — a point in a dense region does
real work, a point in open space bails out immediately. The default static schedule
splits *iteration counts* equally, which gives threads very unequal *work*, and they
idle at the barrier.

### 4.2 Two executor threads, not one per core

`LimoWrapper.cpp`:

```cpp
rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
```

The node has exactly two `MutuallyExclusive` callback groups (LiDAR and IMU), so at
most two callbacks can ever run at once. Every extra executor thread can never pick up
work — it only wakes, contends for the executor mutex, and migrates, stealing cores
from the OpenMP regions inside the LiDAR callback. This is a large part of the
1.30 → 0.83 core drop.

### 4.3 Remaining OpenMP regions

| Location | Work |
|---|---|
| `Localizer.cpp:379` | scan → world transform |
| `Localizer.cpp:402` | (debug) final cloud assembly |
| `Localizer.cpp:591` | `calculate_H` — per-match Jacobian rows |
| `Localizer.cpp:846` | sweep max-timestamp reduction |
| `Localizer.cpp:880` | per-point deskew |
| `Mapper.cpp:87` | kNN + plane fit per point |

All six are embarrassingly parallel over points or matches. That is not a coincidence
any more — it is the shape the code was pushed into.

---

## 5. Where the time goes now

Per the performance board, on the shipped config:

```
Preprocess (filters)    NaN removal, crop box, distance/FoV/rate filter
Sweep copy + max t      O(n), parallel reduction
Deskew                  IMU prior + per-point motion compensation
Voxel grid              PCL downsample                       <- single-threaded, PCL
KF update (total)       iterated update
  - match (kNN)         ~4.68 ms   <- still the largest single item
  - jacobian            calculate_H
Transform to world      scan -> world
Map insert              ~0.26 ms   <- effectively free now
```

`match` is where the remaining headroom is, and it is the stage a GPU helps most.

---

## 6. What this means for a CUDA implementation

The commit was written with a device backend in mind. Concretely:

### 6.1 The seam already exists

Add a `CudaHashGridMapBackend : IMapBackend` and register it in `makeMapBackend()`.
`Mapper`, `Localizer`, and the KF see no change. `params.yaml` picks it with
`backend: cuda_hashgrid`. The octree stays available as the reference peer for
correctness diffing.

### 6.2 The map layout is already a device layout

`HashGrid`'s state is four flat arrays of POD — `table_` (20-byte `Slot`), `used_`,
`pool_` (`Vector3f`), `cursor_`. No pointers, no per-node allocation, integer keys.
That is uploadable verbatim, and it was chosen for that reason.

Porting notes:

- **Insert becomes lock-free.** The probe is `atomicCAS` on a key/occupancy word; the
  point write is `atomicAdd` on `Slot::count` bounded by `max_points_`, and the
  ring-overwrite path is `atomicInc` on `cursor_`. Races between two points landing in
  the same voxel in the same scan are benign — the map is an approximation of the scene,
  and which of two near-coincident points wins does not matter.
- **`reserveFor()`/`rehash()` must move off the critical path.** On device, either
  over-provision the table for the worst-case voxel count and never rehash within a
  scan, or rehash between scans. Note that `pool_` offsets survive a rehash by design,
  so only the slot table needs rebuilding.
- **Consider SoA for `pool_`.** `Vector3f` AoS gives a warp handling one voxel a
  strided load. Splitting into `px[] / py[] / pz[]` makes it coalesced. Because
  `VOXEL_CAPACITY` is a compile-time constant and each voxel owns a contiguous 20-slot
  block, this is a mechanical change — the offset arithmetic does not move.
- **`VOXEL_CAPACITY = 20` is the knob that makes fixed-size device allocation possible.**
  Keep it compile-time.

### 6.3 kNN is the kernel worth writing first

One thread per query point. `KnnHeap` at k=5 is 5 floats + 5 `Vector3f` — it fits in
registers, and the insertion sort has a fixed trip count. **`MAX_KNN = 16` is the
register-pressure ceiling to watch**: if occupancy suffers, template the heap on `k`
rather than raising the cap.

Warp divergence lives in two places, both known:

- The `cellMinDistSq` early-`continue` prune (§2.4) is divergent by construction. It is
  still worth keeping — it is what makes 27 cells affordable — but measure with and
  without on device, since the CPU's cache-miss savings and the GPU's are not the same
  currency.
- Cells have different occupancies (`s.count` ∈ [0,20]), so per-thread trip counts vary.
  This is the GPU analogue of §4.1's load imbalance; the CPU fixed it with
  `schedule(dynamic,64)`, the GPU wants either a persistent-threads scheme or a
  warp-per-query cooperative scan.

### 6.4 Plane fitting is already a device function

`estimate_plane()` is now a fixed 3×3 `AtA` accumulation plus an LDLT solve with no
allocation and no dynamic sizing. It can be marked `__device__` more or less as-is
(Eigen fixed-size types support device code; alternatively write the closed-form 3×3
solve). **Port the finite/degenerate-norm check verbatim** — the NaN-accepted-as-perfect-plane
failure in §3.2 is worse on device, where it is harder to see.

Fuse it into the kNN kernel: the neighbours are already in registers, so writing them
out and reading them back for a separate plane kernel is pure loss.

### 6.5 Deskew and transform are free wins, taken second

Both loops are per-point independent. Deskew's only lookup is
`binary_search_tailored()` over `frames`, which is small (tens to low hundreds of IMU
states) — put it in shared or constant memory. `extract_point_time()` is a pure
function of the point's own field (§3.3), so **no sort, no ordering constraint, no
inter-thread dependency**. The world-frame transform at `Localizer.cpp:379` should be
fused into the same kernel rather than launched separately.

### 6.6 Don't move `H` across the bus

`calculate_H` builds an `N × 12` matrix (N up to `MAX_NUM_MATCHES`) and the KF then
consumes it to form a 12×12 system. Copying `N × 12` doubles back to host per KF
iteration would eat the entire win.

Instead, accumulate `Hᵀ H` (12×12) and `Hᵀ h` (12×1) on device via reduction and
transfer **only those**. Per KF iteration that is 156 doubles instead of ~N×12. The
per-match Jacobian math at `Localizer.cpp:592-617` is already branch-light and
per-match independent; the only conditional is `estimate_extrinsics`, which is a
launch-time constant and should be a template parameter, not a runtime branch.

### 6.7 The transfer budget

The ideal steady state per scan is:

```
host -> device:  raw sweep (already there if zero_copy intake is used)
                 IMU frames (small)
device:          deskew -> transform -> knn+plane -> H^T H / H^T h reduction
                 ... repeated per KF iteration ...
                 map insert
device -> host:  12x12 + 12x1 per KF iteration; nothing else
```

The map **never** comes back. `Matches` never comes back. Note that `config.debug`
paths (`original_scan`, `deskewed_scan`, `this->matches`) do force device→host copies
— they are already gated behind `debug`, and `params.yaml` now ships with
`debug: false` for exactly this reason.

### 6.8 How to validate the port

1. Same bag, `backend: octree` vs `backend: hashgrid` vs `backend: cuda_hashgrid`.
2. Compare **stage times**, not scan time (§1).
3. Compare kNN recall against the octree's results — both go through `KnnHeap`, so the
   orderings are directly comparable (§3.1). Reproduce the recall table in §2.5.
4. Reproduce the incremental, per-scan insert pattern. A batch build measures the wrong
   thing (§2.1).

---

## 7. Files touched

| File | Role |
|---|---|
| [`Objects/MapTypes.hpp`](../include/fast_limo/Objects/MapTypes.hpp) | `MAX_KNN`, `KnnResult`, `KnnHeap` — shared by all backends |
| [`Objects/HashGrid.hpp`](../include/fast_limo/Objects/HashGrid.hpp) | the new map structure |
| [`Objects/MapBackend.hpp`](../include/fast_limo/Objects/MapBackend.hpp) | `IMapBackend` + factory — the CUDA seam |
| [`Objects/Octree.hpp`](../include/fast_limo/Objects/Octree.hpp) | private heap removed, now shares `KnnHeap`; `knn()` signature |
| [`Objects/Plane.cpp/.hpp`](../include/fast_limo/Objects/Plane.cpp) | normal equations, no allocation, degenerate-fit rejection |
| [`Modules/Mapper.cpp/.hpp`](../include/fast_limo/Modules/Mapper.cpp) | backend indirection, stack kNN results, dynamic schedule |
| [`Modules/Localizer.cpp/.hpp`](../include/fast_limo/Modules/Localizer.cpp) | `StageTimes`/`ScopedTimer`, sort removal, inline 1 Hz debug |
| [`IKFoM/use-ikfom.cpp`](../include/IKFoM/use-ikfom.cpp) | match/jacobian timing split |
| [`Utils/Config.hpp`](../include/fast_limo/Utils/Config.hpp) | `backend`, `HashGrid` settings |
| [`src/LimoWrapper.cpp`](../src/LimoWrapper.cpp) | defensive parameter reads, 2-thread executor |
| [`config/params.yaml`](../config/params.yaml) | backend selection + measured tuning tables |
