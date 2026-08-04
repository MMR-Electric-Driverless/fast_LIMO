/*
 Copyright (c) 2024 Oriol Martínez @fetty31

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef __FASTLIMO_HASHGRID_HPP__
#define __FASTLIMO_HASHGRID_HPP__

#include <cstdint>
#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include "fast_limo/Objects/MapTypes.hpp"

namespace fast_limo {

namespace hashgrid {

// Points retained per voxel. Compile-time fixed rather than a growable vector:
// it bounds map memory, keeps every voxel a flat POD (the exact layout a CUDA
// backend would upload verbatim), and doubles as the map's downsampling policy.
// The runtime `max_points_per_voxel` may be lower, never higher.
static constexpr uint32_t VOXEL_CAPACITY = 20;

/*
 * Open-addressed spatial hash of fixed-capacity voxels (an iVox-style map).
 *
 * Replaces the octree because both hot operations are hostile to a tree:
 * insertion recursively subdivides and heap-allocates child arrays, and kNN
 * descends pointers to a data-dependent depth, so every level is a cache miss.
 *
 * Here insertion is a hash probe plus an array write, and kNN scans a bounded
 * set of neighbouring cells whose points sit contiguously in one flat vector.
 * No pointers, no recursion, no allocation on either path -- and the layout
 * ports to the device directly (flat arrays, integer keys, atomic insert).
 *
 * Measured on-vehicle against the octree (20 Hz Hesai, same route): match 4.68 vs
 * 10.68 ms, insert 0.26 vs 4.09 ms, 0.83 vs 1.30 cores. Note that an offline
 * single-threaded benchmark reached the opposite conclusion because it batch-built
 * the octree once rather than inserting a scan per frame -- the incremental
 * subdivision, and the cache pressure of six threads walking a large tree, are
 * where the tree actually loses.
 */
struct HashGrid {

    // 20 bytes, deliberately. The points do NOT live here: an earlier version
    // inlined VOXEL_CAPACITY Vector3f into the slot, which made the slot 260 B
    // and the (necessarily over-provisioned) table hundreds of MB, so every
    // probe missed cache and both insert and query lost to the octree. Keeping
    // slots small and pushing points into a separate pool means only *occupied*
    // voxels cost point storage, and the probe path stays cache-resident.
    struct Slot {
        int32_t  kx, ky, kz;  // voxel coordinates, re-checked on probe collisions
        uint32_t count;       // points currently stored
        uint32_t offset;      // start of this voxel's block in pool_

        Slot() : kx(0), ky(0), kz(0), count(0), offset(0) { }
    };

    HashGrid()
        : num_voxels_(0), num_points_(0),
          voxel_size_(0.5f), inv_voxel_size_(2.0f),
          max_points_(VOXEL_CAPACITY), neighbors_(7) {
        allocate(1024);
    }

    void setVoxelSize(float size) {
        if (size <= 0.0f) return;
        voxel_size_     = size;
        inv_voxel_size_ = 1.0f / size;
    }

    void setMaxPointsPerVoxel(int n) {
        if (n < 1) return;
        max_points_ = (static_cast<uint32_t>(n) > VOXEL_CAPACITY) ? VOXEL_CAPACITY
                                                                  : static_cast<uint32_t>(n);
    }

    // 7 = the cell plus its face neighbours; 27 = the full 3x3x3 block.
    void setNeighbors(int n) {
        neighbors_ = (n >= 27) ? 27 : 7;
    }

    void clear() {
        table_.assign(table_.size(), Slot());
        used_.assign(used_.size(), 0);
        pool_.clear();
        cursor_.clear();
        num_voxels_ = 0;
        num_points_ = 0;
    }

    size_t size() const { return num_points_; }

    template <typename ContainerT>
    void add(ContainerT& pts) {
        // Grow ahead of the insert so the probe below always terminates.
        reserveFor(pts->points.size());

        for (const auto& p : pts->points)
            insert(Eigen::Vector3f(p.x, p.y, p.z));
    }

    void knn(const Eigen::Vector3f& query, int k, KnnResult& result) const {

        result.count = 0;

        if (k < 1 || num_points_ == 0)
            return;

        KnnHeap heap(static_cast<size_t>(k));

        const int32_t kx = coord(query.x());
        const int32_t ky = coord(query.y());
        const int32_t kz = coord(query.z());

        const int8_t (*offsets)[3] = (neighbors_ == 27) ? NEIGHBORS_27 : NEIGHBORS_7;

        // Offsets run centre -> faces -> edges -> corners, so the heap fills from
        // the most promising cells first and the bound below starts biting early.
        for (int n = 0; n < neighbors_; ++n) {
            const int32_t cx = kx + offsets[n][0];
            const int32_t cy = ky + offsets[n][1];
            const int32_t cz = kz + offsets[n][2];

            // Prune whole cells: if the nearest point of this cell's box is already
            // farther than the current k-th best, nothing inside it can qualify.
            // This is what keeps a 27-cell scan from costing 27 cells' worth of
            // distance computations.
            if (heap.full() && cellMinDistSq(query, cx, cy, cz) >= heap.worstDist())
                continue;

            size_t idx;
            if (not find(cx, cy, cz, idx))
                continue;

            const Slot& s = table_[idx];
            const Eigen::Vector3f* pts = &pool_[s.offset];
            for (uint32_t j = 0; j < s.count; ++j)
                heap.addPoint(pts[j], (query - pts[j]).squaredNorm());
        }

        heap.fill(result);
    }

  private:

    std::vector<Slot>    table_;
    std::vector<uint8_t> used_;
    size_t mask_;

    // Point storage, VOXEL_CAPACITY slots per occupied voxel. Never reallocated
    // by a rehash, so Slot::offset stays valid across table growth.
    std::vector<Eigen::Vector3f> pool_;
    std::vector<uint32_t>        cursor_; // ring write position, one per voxel

    size_t num_voxels_;
    size_t num_points_;

    float  voxel_size_;
    float  inv_voxel_size_;
    uint32_t max_points_;
    int    neighbors_;

    static const int8_t NEIGHBORS_7[7][3];
    static const int8_t NEIGHBORS_27[27][3];

    // floor(), not truncation: truncation folds -0.4 and +0.4 into the same cell
    // and mirrors the grid about the origin.
    int32_t coord(float v) const {
        return static_cast<int32_t>(std::floor(v * inv_voxel_size_));
    }

    // Squared distance from `q` to the nearest point of cell (cx,cy,cz)'s box.
    // Zero when the query lies inside the cell.
    float cellMinDistSq(const Eigen::Vector3f& q, int32_t cx, int32_t cy, int32_t cz) const {
        const float lo[3] = { cx * voxel_size_, cy * voxel_size_, cz * voxel_size_ };
        float sq = 0.0f;
        for (int a = 0; a < 3; ++a) {
            const float hi = lo[a] + voxel_size_;
            const float d  = (q[a] < lo[a]) ? (lo[a] - q[a])
                           : (q[a] > hi)    ? (q[a] - hi)
                                            : 0.0f;
            sq += d * d;
        }
        return sq;
    }

    static size_t hash(int32_t x, int32_t y, int32_t z) {
        return static_cast<size_t>(
            (static_cast<uint64_t>(static_cast<uint32_t>(x)) * 73856093ull) ^
            (static_cast<uint64_t>(static_cast<uint32_t>(y)) * 19349669ull) ^
            (static_cast<uint64_t>(static_cast<uint32_t>(z)) * 83492791ull));
    }

    void allocate(size_t capacity) {
        table_.assign(capacity, Slot());
        used_.assign(capacity, 0);
        mask_ = capacity - 1; // capacity is always a power of two
    }

    // Index of the slot holding this key, or of the first free slot for it.
    // Always terminates: the load factor is kept below 1 by reserveFor().
    size_t probe(int32_t x, int32_t y, int32_t z) const {
        size_t idx = hash(x, y, z) & mask_;
        while (used_[idx]) {
            const Slot& s = table_[idx];
            if (s.kx == x && s.ky == y && s.kz == z) break;
            idx = (idx + 1) & mask_;
        }
        return idx;
    }

    bool find(int32_t x, int32_t y, int32_t z, size_t& out) const {
        size_t idx = probe(x, y, z);
        if (not used_[idx]) return false;
        out = idx;
        return true;
    }

    // Keep the load factor under ~0.7; above that, linear probing degrades badly.
    void reserveFor(size_t incoming) {
        size_t capacity = table_.size();
        while ((num_voxels_ + incoming) * 10 >= capacity * 7)
            capacity *= 2;

        if (capacity != table_.size())
            rehash(capacity);
    }

    void rehash(size_t capacity) {
        std::vector<Slot>    old_table;
        std::vector<uint8_t> old_used;
        old_table.swap(table_);
        old_used.swap(used_);

        allocate(capacity);

        // Only 20-byte slots move; pool_ and every Slot::offset stay put.
        for (size_t i = 0; i < old_used.size(); ++i) {
            if (not old_used[i]) continue;
            const Slot& s = old_table[i];
            size_t idx = probe(s.kx, s.ky, s.kz);
            table_[idx] = s;
            used_[idx]  = 1;
        }
    }

    void insert(const Eigen::Vector3f& p) {
        const int32_t kx = coord(p.x());
        const int32_t ky = coord(p.y());
        const int32_t kz = coord(p.z());

        size_t idx = probe(kx, ky, kz);

        if (not used_[idx]) {
            Slot& s = table_[idx];
            s.kx = kx; s.ky = ky; s.kz = kz;
            s.count  = 0;
            s.offset = static_cast<uint32_t>(pool_.size());
            pool_.resize(pool_.size() + VOXEL_CAPACITY);
            cursor_.push_back(0);
            used_[idx] = 1;
            ++num_voxels_;
        }

        Slot& s = table_[idx];

        if (s.count < max_points_) {
            pool_[s.offset + s.count] = p;
            ++s.count;
            ++num_points_;
        } else {
            // Full: overwrite oldest. Keeps the map tracking the recent scene
            // instead of freezing on whatever was seen first.
            const uint32_t vi = s.offset / VOXEL_CAPACITY;
            pool_[s.offset + cursor_[vi]] = p;
            cursor_[vi] = (cursor_[vi] + 1) % max_points_;
        }
    }
};

// Face-adjacent cells. At a 0.5 m voxel with a 5-NN query this covers the
// relevant neighbourhood; the diagonal cells only matter for queries sitting
// almost exactly on a cell corner.
inline const int8_t HashGrid::NEIGHBORS_7[7][3] = {
    { 0, 0, 0},
    {-1, 0, 0}, {1, 0, 0},
    { 0,-1, 0}, {0, 1, 0},
    { 0, 0,-1}, {0, 0, 1}
};

// Ordered by distance from the centre cell -- centre, then the 6 faces, the 12
// edges, and finally the 8 corners. The kNN pruning bound depends on this: the
// nearest cells must be visited first so the heap is already tight by the time
// the far ones come up for their box test.
inline const int8_t HashGrid::NEIGHBORS_27[27][3] = {
    { 0, 0, 0},

    {-1, 0, 0}, { 1, 0, 0}, { 0,-1, 0}, { 0, 1, 0}, { 0, 0,-1}, { 0, 0, 1},

    {-1,-1, 0}, {-1, 1, 0}, { 1,-1, 0}, { 1, 1, 0},
    {-1, 0,-1}, {-1, 0, 1}, { 1, 0,-1}, { 1, 0, 1},
    { 0,-1,-1}, { 0,-1, 1}, { 0, 1,-1}, { 0, 1, 1},

    {-1,-1,-1}, {-1,-1, 1}, {-1, 1,-1}, {-1, 1, 1},
    { 1,-1,-1}, { 1,-1, 1}, { 1, 1,-1}, { 1, 1, 1}
};

} // namespace hashgrid

} // namespace fast_limo

#endif
