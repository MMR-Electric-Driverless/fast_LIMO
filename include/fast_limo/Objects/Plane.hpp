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

#ifndef __FASTLIMO_PLANE_HPP__
#define __FASTLIMO_PLANE_HPP__

#include "fast_limo/Common.hpp"
#include "fast_limo/Utils/Config.hpp"

class fast_limo::Plane{

    public:

        // `pts`/`sq_dists` are the n nearest map points and their squared
        // distances, sorted ascending. Taken as raw pointers so the caller can
        // pass stack buffers: this is built once per kNN query (~30k per scan),
        // so no allocation may happen along this path.
        Plane(const Eigen::Vector3f* pts, const float* sq_dists, int n,
              Config::iKFoM::Mapping* config_ptr);
        Plane() = default;

        Eigen::Vector4f get_normal();
        bool good_fit();

        float dist2plane(const Eigen::Vector3f&) const;
        float dist2plane(const PointType&) const;

        bool on_plane(const Eigen::Vector3f&);
        bool on_plane(const PointType&);
        
        bool enough_points(int n);
        bool close_enough(const float* sq_dists, int n);

    private:
        Eigen::Vector4f n_ABCD; // plane normal vector
        bool is_plane;

        Config::iKFoM::Mapping* cfg_ptr;

        void fit_plane(const Eigen::Vector3f* pts, int n);

        bool estimate_plane(const Eigen::Vector3f* pts, int n, Eigen::Vector4f& out);

        bool plane_eval(const Eigen::Vector4f&, const Eigen::Vector3f* pts, int n, const float&);

};

#endif
