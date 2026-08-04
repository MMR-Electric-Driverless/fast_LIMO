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

#include "fast_limo/Objects/Plane.hpp"

// class fast_limo::Plane
    // public

        fast_limo::Plane::Plane(const Eigen::Vector3f* pts, const float* sq_dists, int n,
                                Config::iKFoM::Mapping* config_ptr) : is_plane(false),
                                                                 cfg_ptr(config_ptr) {
            if(not enough_points(n)) return;
            if(not close_enough(sq_dists, n)) return;

            // Get normal vector of plane between p points
            this->fit_plane(pts, n);
        }

        Eigen::Vector4f fast_limo::Plane::get_normal(){
            return this->n_ABCD;
        }

        bool fast_limo::Plane::good_fit(){
            return this->is_plane;
        }

        bool fast_limo::Plane::enough_points(int n){
            return this->is_plane = n >= cfg_ptr->NUM_MATCH_POINTS;
        }

        bool fast_limo::Plane::close_enough(const float* sq_dists, int n){
            if(n < 1) return this->is_plane = false;
            // sq_dists is sorted ascending, so the last entry is the farthest.
            return this->is_plane = sq_dists[n-1] < cfg_ptr->MAX_DIST_PLANE;
        }

        float fast_limo::Plane::dist2plane(const Eigen::Vector3f& p) const {
            return n_ABCD(0) * p(0) + n_ABCD(1) * p(1) + n_ABCD(2) * p(2) + n_ABCD(3);
        }

        float fast_limo::Plane::dist2plane(const PointType& p) const {
            return n_ABCD(0) * p.x + n_ABCD(1) * p.y + n_ABCD(2) * p.z + n_ABCD(3);
        }

        bool fast_limo::Plane::on_plane(const Eigen::Vector3f& p) {
            if(not this->is_plane) return false;
            return std::fabs(this->dist2plane(p)) < cfg_ptr->PLANE_THRESHOLD;
        }

        bool fast_limo::Plane::on_plane(const PointType& p) {
            if(not this->is_plane) return false;
            return std::fabs(this->dist2plane(p)) < cfg_ptr->PLANE_THRESHOLD;
        }
    
    // private

        void fast_limo::Plane::fit_plane(const Eigen::Vector3f* pts, int n){
            // Estimate plane
            if(not this->estimate_plane(pts, n, this->n_ABCD)){
                this->is_plane = false;
                return;
            }

            this->is_plane = this->plane_eval(n_ABCD, pts, n, cfg_ptr->PLANE_THRESHOLD);
        }

        bool fast_limo::Plane::estimate_plane(const Eigen::Vector3f* pts, int n, Eigen::Vector4f& out){
            // Least-squares fit of A*x = b with b = -1, i.e. the plane
            // x(0)*px + x(1)*py + x(2)*pz + 1 = 0.
            //
            // Solved through the normal equations (A^T A) x = A^T b accumulated in
            // place rather than a QR of a dynamically-sized A. Both are least-squares
            // solutions of the same system, but A^T A is a fixed 3x3 that stays in
            // registers, whereas the dynamic A, b and the QR's internals were four
            // separate heap allocations per call.
            Eigen::Matrix3f AtA = Eigen::Matrix3f::Zero();
            Eigen::Vector3f Atb = Eigen::Vector3f::Zero();

            for (int j = 0; j < n; j++) {
                const Eigen::Vector3f& a = pts[j];
                AtA.noalias() += a * a.transpose();
                Atb -= a; // b(j) == -1
            }

            Eigen::Vector3f normvec = AtA.ldlt().solve(Atb);

            // Degenerate neighbourhoods (collinear or coincident map points) leave AtA
            // singular, and the solve then yields inf/NaN. That must be rejected here:
            // plane_eval() compares with `>`, which is false for NaN, so a NaN normal
            // would otherwise be accepted as a perfect plane.
            const float norm = normvec.norm();
            if (not std::isfinite(norm) || norm < 1e-6f)
                return false;

            out(0) = normvec(0) / norm;
            out(1) = normvec(1) / norm;
            out(2) = normvec(2) / norm;
            out(3) = 1.0f / norm;

            return true;
        }

        bool fast_limo::Plane::plane_eval(const Eigen::Vector4f& n, const Eigen::Vector3f* pts,
                                          int num, const float& thres){
            for (int j = 0; j < num; j++) {
                float res = n(0) * pts[j].x() + n(1) * pts[j].y() + n(2) * pts[j].z() + n(3);
                if (fabs(res) > thres) return false;
            }

            return true;
        }
