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

#ifndef __FASTLIMO_MAPBACKEND_HPP__
#define __FASTLIMO_MAPBACKEND_HPP__

#include <memory>
#include <string>

#include "fast_limo/Common.hpp"
#include "fast_limo/Utils/Config.hpp"
#include "fast_limo/Objects/MapTypes.hpp"
#include "fast_limo/Objects/Octree.hpp"
#include "fast_limo/Objects/HashGrid.hpp"

namespace fast_limo {

/*
 * The map, behind an interface.
 *
 * Mapper only ever asks the map two things -- "insert this scan" and "give me
 * the k nearest points to this position" -- and together those are ~60% of a
 * scan's compute. Putting them behind a Bridge lets a different structure be
 * swapped in and A/B'd against the original on the same bag, and gives a CUDA
 * peer (device-resident map, only query results crossing back) somewhere to
 * land later without touching the localizer.
 *
 * Both peers return results through the shared KnnHeap, so their output is
 * ordered identically and any accuracy difference is the structure's, not the
 * comparison's.
 */
class IMapBackend {
  public:
    virtual ~IMapBackend() = default;

    virtual void configure(const Config::iKFoM::Mapping& cfg) = 0;

    virtual size_t size() const = 0;

    virtual void add(pcl::PointCloud<PointType>::Ptr& pc) = 0;

    // `k` is clamped to MAX_KNN by the implementations.
    virtual void knn(const Eigen::Vector3f& query, int k, KnnResult& result) = 0;

    virtual const char* name() const = 0;
};


// Incremental octree: the original fast-LIMO map, kept as the reference peer.
class OctreeMapBackend : public IMapBackend {
  public:
    void configure(const Config::iKFoM::Mapping& cfg) override {
        octree_.setBucketSize(cfg.octree.bucket_size);
        octree_.setDownsample(cfg.octree.downsampling);
        octree_.setMinExtent(cfg.octree.min_extent);
    }

    size_t size() const override { return octree_.num_points_; }

    void add(pcl::PointCloud<PointType>::Ptr& pc) override {
        if (octree_.num_points_ < 1) octree_.initialize(pc);
        else                         octree_.update(pc);
    }

    void knn(const Eigen::Vector3f& query, int k, KnnResult& result) override {
        octree_.knn(query, k, result);
    }

    const char* name() const override { return "octree"; }

  private:
    octree::Octree octree_;
};


// iVox-style spatial hash. See HashGrid.hpp for why this shape was chosen.
class HashGridMapBackend : public IMapBackend {
  public:
    void configure(const Config::iKFoM::Mapping& cfg) override {
        grid_.setVoxelSize(cfg.hash_grid.voxel_size);
        grid_.setMaxPointsPerVoxel(cfg.hash_grid.max_points_per_voxel);
        grid_.setNeighbors(cfg.hash_grid.neighbors);
    }

    size_t size() const override { return grid_.size(); }

    void add(pcl::PointCloud<PointType>::Ptr& pc) override { grid_.add(pc); }

    void knn(const Eigen::Vector3f& query, int k, KnnResult& result) override {
        grid_.knn(query, k, result);
    }

    const char* name() const override { return "hashgrid"; }

  private:
    hashgrid::HashGrid grid_;
};


// Single place where the backend choice is made. Unknown names fall back to the
// octree rather than failing, so a stale config can't take the vehicle down.
inline std::unique_ptr<IMapBackend> makeMapBackend(const std::string& name) {
    if (name == "hashgrid")
        return std::unique_ptr<IMapBackend>(new HashGridMapBackend());

    if (name != "octree")
        std::cout << "FAST_LIMO::Mapper: unknown map backend '" << name
                  << "', falling back to 'octree'.\n";

    return std::unique_ptr<IMapBackend>(new OctreeMapBackend());
}

} // namespace fast_limo

#endif
