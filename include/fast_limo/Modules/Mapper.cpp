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

#include "fast_limo/Modules/Mapper.hpp"

// class fast_limo::Mapper
    // public

        Mapper::Mapper() : last_map_time(-1.), num_threads_(1){

            // Init cfg values
            this->config.NUM_MATCH_POINTS = 5;
            this->config.MAX_NUM_PC2MATCH = 1.e+4;
            this->config.MAX_DIST_PLANE   = 2.0;
            this->config.PLANE_THRESHOLD  = 5.e-2;
            this->config.backend          = "octree";

            // set_config() replaces this once the real config arrives, but the
            // singleton may be queried before then.
            this->backend_ = makeMapBackend(this->config.backend);
        }

        void Mapper::set_num_threads(int n){
            if(n < 1) return;
            this->num_threads_ = n;
        }

        void Mapper::set_config(const Config::iKFoM::Mapping& cfg){
            this->config = cfg;

            // The kNN search returns at most MAX_KNN neighbours. Asking for more
            // would silently starve every plane fit (enough_points() could never be
            // satisfied), so clamp loudly instead.
            if(this->config.NUM_MATCH_POINTS > static_cast<int>(MAX_KNN)){
                std::cout << "FAST_LIMO::Mapper: NUM_MATCH_POINTS (" << this->config.NUM_MATCH_POINTS
                          << ") exceeds the kNN limit of " << MAX_KNN << ", clamping.\n";
                this->config.NUM_MATCH_POINTS = static_cast<int>(MAX_KNN);
            }

            // Build the selected map structure and hand it its parameters
            this->backend_ = makeMapBackend(this->config.backend);
            this->backend_->configure(this->config);

            std::cout << "FAST_LIMO::Mapper: using '" << this->backend_->name()
                      << "' map backend.\n";
        }
                
        bool Mapper::exists(){
            return this->backend_->size() > 0;
        }

        int Mapper::size(){
            return static_cast<int>(this->backend_->size());
        }

        double Mapper::last_time(){
            return this->last_map_time;
        }

        Matches Mapper::match(State s, pcl::PointCloud<PointType>::Ptr& pc){

            if(not this->exists()) return matches;

            int N0 = (pc->points.size() > config.MAX_NUM_PC2MATCH) ? pc->points.size() - config.MAX_NUM_PC2MATCH : 0;

            Matches init_matches;
            init_matches.resize(pc->points.size()-N0);

            // Dynamic schedule: kNN cost per point varies by an order of magnitude
            // (a point in a dense octant descends deep, one in open space bails out
            // immediately). With the default static split into equal iteration counts
            // the threads get very unequal *work* and idle at the barrier.
            #pragma omp parallel for num_threads(this->num_threads_) schedule(dynamic, 64)
            for(size_t i= 0; i < pc->points.size()-N0; i++){
                
                Eigen::Vector4f bl4_point(pc->points[i].x, pc->points[i].y, pc->points[i].z, 1.); // base link 4d point
                Eigen::Vector4f global_point = s.get_RT() * bl4_point;                            // global 4d point == [x', y', z', 1.0]
                Match match = this->match_plane(global_point, bl4_point);                         // point-to-plane match

                init_matches[i] = match; 
            }

            Matches chosen_matches;
            for(size_t j = 0; j < init_matches.size(); j++){
                if(init_matches[j].lisanAlGaib())
                    chosen_matches.push_back(init_matches[j]); // if match is chosen, push it
            }

            this->matches = chosen_matches; // save matches for next iter
            return chosen_matches;
        }
        
        void Mapper::add(pcl::PointCloud<PointType>::Ptr& pc, double time){
            if(pc->points.size() < 1) return;

            this->backend_->add(pc);

            this->last_map_time = time;
        }

    // private

        Match Mapper::match_plane(Eigen::Vector4f& p, Eigen::Vector4f& p_local) {

            // Find k nearest points. The result buffer is fixed-capacity and lives on
            // the stack: this runs once per point per KF iteration (~30k times per
            // scan, across every OpenMP thread), so any allocation here would both
            // dominate the search cost and serialise the threads on the malloc arena.
            KnnResult near_points;

            this->backend_->knn(p.head<3>(), this->config.NUM_MATCH_POINTS, near_points);

            // Construct a plane fitting between them
            return Match( p.head(3), p_local.head(3),
                          Plane (near_points.points, near_points.dists, near_points.count, &config) );
        }