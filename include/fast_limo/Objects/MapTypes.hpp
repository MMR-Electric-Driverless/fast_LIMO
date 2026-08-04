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

#ifndef __FASTLIMO_MAPTYPES_HPP__
#define __FASTLIMO_MAPTYPES_HPP__

#include <cstddef>
#include <limits>
#include <Eigen/Dense>

namespace fast_limo {

// Upper bound on k for any map backend's kNN. Every shipped config uses
// NUM_MATCH_POINTS == 5, so a small fixed cap lets both the search heap and its
// results live on the stack -- the query runs ~30k times per scan across every
// OpenMP thread, and a single allocation there costs more than the search.
static constexpr size_t MAX_KNN = 16;

// Result of a kNN query. `dists` are SQUARED distances, sorted ascending and
// aligned with `points`. Fixed capacity so callers can keep it on the stack.
struct KnnResult {
    Eigen::Vector3f points[MAX_KNN];
    float           dists[MAX_KNN];
    int             count;

    KnnResult() : count(0) { }
};

// Bounded collection of the k nearest candidates seen so far. Insertion sort
// rather than a real heap: k is 5, so the linear shift beats the branchy
// sift-down and keeps the output sorted for free (Plane::close_enough() relies
// on the last element being the farthest).
//
// Shared by every map backend so they all return identically ordered results,
// which is what makes an octree-vs-hashgrid A/B meaningful.
struct KnnHeap {
    size_t capacity;
    size_t count;
    float           dists[MAX_KNN];
    Eigen::Vector3f points[MAX_KNN];

    explicit KnnHeap(size_t capacity_)
        : capacity(capacity_ < MAX_KNN ? capacity_ : MAX_KNN), count(0) { }

    bool full() const { return count == capacity; }

    float worstDist() const {
        return full() ? dists[count-1] : std::numeric_limits<float>::max();
    }

    void addPoint(const Eigen::Vector3f& p, float dist) {
        if (full() and dist >= dists[count-1])
            return;

        if (count < capacity)
            ++count;

        int i = static_cast<int>(count) - 1;
        while (i > 0 && dists[i-1] > dist) {
            dists[i]  = dists[i-1];
            points[i] = points[i-1];
            --i;
        }

        dists[i]  = dist;
        points[i] = p;
    }

    void fill(KnnResult& out) const {
        out.count = static_cast<int>(count);
        for (size_t i = 0; i < count; ++i) {
            out.points[i] = points[i];
            out.dists[i]  = dists[i];
        }
    }
};

} // namespace fast_limo

#endif
