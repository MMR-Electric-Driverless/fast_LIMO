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

#include "ROSutils.hpp"

#ifdef LATENCY_TESTING
#include <chrono>
#include <mmr_base/msg/latency_sample.hpp>
#endif

namespace ros2wrap {

    class LimoWrapper : public rclcpp::Node
    {

        // VARIABLES

        public:
            std::string world_frame;
            std::string body_frame;
            std::string lidar_frame;

            bool debug;
            bool publish_tf;
            bool zero_copy;
            bool barq;

        private:
                // subscribers
            rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr      lidar_sub_;
            #ifdef ENABLE_ZERO_COPY
            rclcpp::Subscription<mmr_base::msg::BoundedPointcloud>::SharedPtr   bounded_lidar_sub_;
            #endif
            rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr              imu_sub_;

                /* Held as a member because the BARQ transport has no subscription
                   to carry it: the poll timer and the writer health check go into
                   the SAME MutuallyExclusive group the lidar subscription would
                   have used, so a frame being processed cannot be re-entered by
                   the next poll and the two-thread executor in main() still has
                   exactly one runnable callback per group. */
            rclcpp::CallbackGroup::SharedPtr lidar_cbg_;

            #ifdef ENABLE_BARQ
                // BARQ transport
            std::unique_ptr<BARQ::Reader> barq_reader_;
            rclcpp::TimerBase::SharedPtr  barq_poll_timer_;
            rclcpp::TimerBase::SharedPtr  barq_health_timer_;
            rclcpp::TimerBase::SharedPtr  barq_connect_timer_;
            std::string barq_topic_             = "/lidar_points";
            size_t      barq_max_size_          = 0;
            int         barq_retry_delay_ms_    = 100;
            int         barq_max_retries_       = 100;   // x100 ms = 10 s, see loadBarqConfig
            double      barq_polling_rate_ms_   = 1.0;
            int         barq_writer_timeout_ms_ = 1000;
            int         barq_retries_           = 0;
            #endif

                // main publishers
            rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub;
            rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr       state_pub;
#ifdef LATENCY_TESTING
            rclcpp::Publisher<mmr_base::msg::LatencySample>::SharedPtr   latency_sample_pub_;
            uint32_t latency_seq_ = 0;
            int64_t  latency_t_in_ = 0;
            static int64_t latencyMonotonicNs() {
                return std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
            }
#endif

                // debug publishers
            rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr orig_pub;
            rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr desk_pub;
            rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr match_pub;
            rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr finalraw_pub;
            rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr body_pub;
            rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr map_bb_pub;
            rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr match_points_pub;

                // TF 
            std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

        // FUNCTIONS

        public:

            LimoWrapper() : Node("fast_limo", 
                                rclcpp::NodeOptions()
                                .automatically_declare_parameters_from_overrides(true) ) 
                {

                    // Declare the one and only Localizer and Mapper objects
                    fast_limo::Localizer& LOC = fast_limo::Localizer::getInstance();
                    [[maybe_unused]] fast_limo::Mapper& MAP = fast_limo::Mapper::getInstance();

                    // Load config
                    fast_limo::Config config;
                    this->loadConfig(&config);

                    rclcpp::Parameter tf_pub = this->get_parameter("frames.tf_pub");
                    this->publish_tf = tf_pub.as_bool();

                    rclcpp::Parameter debug_p = this->get_parameter("debug");
                    this->debug = debug_p.as_bool();

                    rclcpp::Parameter zero_copy_p = this->get_parameter("zero_copy");
                    this->zero_copy = zero_copy_p.as_bool();

                    /* Third transport. Optional key, so a params file predating it
                       still starts on ROS2 -- get_parameter would throw here, since
                       the node declares from overrides only (see loadConfig). */
                    this->barq = this->has_parameter("BARQ_enabled")
                            ? this->get_parameter("BARQ_enabled").as_bool() : false;

                    /* REFUSED, not silently resolved in favour of one of them.
                       These are two different transports for the same cloud and
                       picking one behind the operator's back is precisely how an
                       A/B ends up with two arms that were secretly the same arm. */
                    if(this->barq && this->zero_copy){
                        RCLCPP_ERROR_STREAM(this->get_logger(), "\n-------------------------------------------------------------------\n"
                                            << "FAST_LIMO::FATAL ERROR: BARQ_enabled and zero_copy are BOTH true.\n"
                                            << "          They are alternative transports for the same input cloud;\n"
                                            << "          enable exactly one (or neither, for plain ROS2).\n"
                                            << "-------------------------------------------------------------------\n"
                                            );
                        throw std::runtime_error("FAST_LIMO::FATAL ERROR: BARQ_enabled and zero_copy are mutually exclusive\n\n");
                    }

                    // Define two callback groups (ensure parallel execution of lidar_callback & imu_callback)
                    rclcpp::SubscriptionOptions lidar_opt, imu_opt;
                    this->lidar_cbg_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
                    lidar_opt.callback_group = this->lidar_cbg_;
                    imu_opt.callback_group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

                    // Set up subscribers
                    auto qos = rclcpp::QoS(rclcpp::SensorDataQoS());
                    // depth is already 5 from the profile, override if needed:
                    // qos.keep_last(10);

                    if(this->barq){         // BARQ, like zero copy, is lidar-only
                    #ifdef ENABLE_BARQ
                        this->loadBarqConfig(config.topics.lidar);
                        this->barqReaderInit();
                    #else
                        RCLCPP_ERROR_STREAM(this->get_logger(), "\n-------------------------------------------------------------------\n"
                                            << "FAST_LIMO::FATAL ERROR: BARQ_enabled is true but fast_limo was built WITHOUT BARQ support!\n"
                                            << "          Rebuild with ENABLE_BARQ=1 (or -DENABLE_BARQ=ON) and the barq package\n"
                                            << "          available, or set BARQ_enabled to false in the config file.\n"
                                            << "-------------------------------------------------------------------\n"
                                            );
                        throw std::runtime_error("FAST_LIMO::FATAL ERROR: BARQ support not compiled in\n\n");
                    #endif
                    }
                    else if(this->zero_copy){    // zero copy just on lidar for now
                    #ifdef ENABLE_ZERO_COPY
                        bounded_lidar_sub_ = this->create_subscription<mmr_base::msg::BoundedPointcloud>(
                                    config.topics.lidar, qos, std::bind(&LimoWrapper::boundedpointcloud_callback, this, std::placeholders::_1), lidar_opt);
                    #else
                        RCLCPP_ERROR_STREAM(this->get_logger(), "\n-------------------------------------------------------------------\n"
                                            << "FAST_LIMO::FATAL ERROR: zero_copy is enabled but fast_limo was built WITHOUT zero copy support!\n"
                                            << "          Rebuild with RMW_FASTRTPS_USE_QOS_FROM_XML=1 (or -DENABLE_ZERO_COPY=ON) and mmr_base\n"
                                            << "          available, or set zero_copy to false in the config file.\n"
                                            << "-------------------------------------------------------------------\n"
                                            );
                        throw std::runtime_error("FAST_LIMO::FATAL ERROR: zero copy support not compiled in\n\n");
                    #endif
                    }
                    else{
                        lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                                    config.topics.lidar, qos, std::bind(&LimoWrapper::pointcloud_callback, this, std::placeholders::_1), lidar_opt);
                    }
                    imu_sub_   = this->create_subscription<sensor_msgs::msg::Imu>(
                                    config.topics.imu, qos, std::bind(&LimoWrapper::imu_callback, this, std::placeholders::_1), imu_opt);
                    
                    // Set up publishers
                    pc_pub      = this->create_publisher<sensor_msgs::msg::PointCloud2>("/fast_limo/pointcloud", 1);
                    state_pub   = this->create_publisher<nav_msgs::msg::Odometry>("/fast_limo/state", 1);
#ifdef LATENCY_TESTING
                    latency_sample_pub_ = this->create_publisher<mmr_base::msg::LatencySample>(
                        "/latency/sample/fast_limo", rclcpp::QoS(rclcpp::KeepLast(100)).reliable());
                    RCLCPP_INFO(this->get_logger(),
                        "LATENCY_TESTING: per-frame samples on /latency/sample/fast_limo");
#endif

                    if(this->debug)
                    {
                    orig_pub     = this->create_publisher<sensor_msgs::msg::PointCloud2>("/fast_limo/original", 1);
                    desk_pub     = this->create_publisher<sensor_msgs::msg::PointCloud2>("/fast_limo/deskewed", 1);
                    match_pub    = this->create_publisher<sensor_msgs::msg::PointCloud2>("/fast_limo/match", 1);
                    finalraw_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("/fast_limo/final_raw", 1);
                    body_pub     = this->create_publisher<nav_msgs::msg::Odometry>("/fast_limo/body", 1);
                    match_points_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>("/fast_limo/match_points", 1);
                    }

                    // Init TF broadcaster
                    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

                    // Initialize Localizer
                    LOC.init(config);
                }
            
            private:
            
            /* ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
               ///////////////////////////////////////             Callbacks            ///////////////////////////////////////////////////////////// 
               ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// */

            /**
             * reference for zero copy:
                # Sensor frame-start timestamp.
                builtin_interfaces/Time stamp

                # Number of valid points (<= DATA_MAX_POINTS).
                uint32 width

                # Fixed-size payload. Only the first `width * POINT_STEP` bytes are valid;
                # the remaining bytes are unused padding required to keep the message a
                # plain (constant-size) type.
                #
                # Per-point layout (26 bytes):
                #
                #   sensor_msgs/PointField[]                 offset  datatype  count
                #   --------------------------------------------------------------------
                #   { name: "x",         offset:  0, datatype: FLOAT32 (7), count: 1 }
                #   { name: "y",         offset:  4, datatype: FLOAT32 (7), count: 1 }
                #   { name: "z",         offset:  8, datatype: FLOAT32 (7), count: 1 }
                #   { name: "intensity", offset: 12, datatype: FLOAT32 (7), count: 1 }
                #   { name: "ring",      offset: 16, datatype: UINT16  (4), count: 1 }
                #   { name: "timestamp", offset: 18, datatype: FLOAT64 (8), count: 1 }
                #                                              total = 26 bytes
                uint8[3407872] data

                # Constants
                uint32 POINT_STEP=26
                uint32 DATA_MAX_POINTS=128000
                uint32 DATA_MAX_BYTES=3407872
             */

            #ifdef ENABLE_ZERO_COPY
            /* Zero copy entry point: the BoundedPointcloud layout is known at compile time,
               so the payload is unpacked by hand (no PointField metadata is transmitted). */
            void boundedpointcloud_callback(const mmr_base::msg::BoundedPointcloud & msg) {
#ifdef LATENCY_TESTING
                /* First statement: after the middleware delivered, before this
                   node touches the payload. fromROStoLimo below is this node's
                   work and belongs in compute, not in delivery. */
                this->latency_t_in_ = latencyMonotonicNs();
#endif

                fast_limo::Localizer& loc = fast_limo::Localizer::getInstance();
                static bool pc_in_good_shape = this->checkPointcloudStructure(loc.get_sensor_type());

                if(not pc_in_good_shape){
                    throw std::runtime_error("FAST_LIMO::FATAL ERROR: invalid pointcloud structure\n\n");
                }

                pcl::PointCloud<PointType>::Ptr pc_ (std::make_shared<pcl::PointCloud<PointType>>());
                this->fromROStoLimo(msg, *pc_);

                this->lidar_callback(pc_, rclcpp::Time(msg.stamp).seconds());
            }
            #endif

            /* Standard entry point */
            void pointcloud_callback(const sensor_msgs::msg::PointCloud2 & msg) {
#ifdef LATENCY_TESTING
                // See boundedpointcloud_callback.
                this->latency_t_in_ = latencyMonotonicNs();
#endif

                fast_limo::Localizer& loc = fast_limo::Localizer::getInstance();
                static bool pc_in_good_shape = this->checkPointcloudStructure(msg, loc.get_sensor_type());

                if(not pc_in_good_shape){
                    throw std::runtime_error("FAST_LIMO::FATAL ERROR: invalid pointcloud structure\n\n");
                }

                pcl::PointCloud<PointType>::Ptr pc_ (std::make_shared<pcl::PointCloud<PointType>>());
                pcl::fromROSMsg(msg, *pc_);

                this->lidar_callback(pc_, rclcpp::Time(msg.header.stamp).seconds());
            }

            #ifdef ENABLE_BARQ
            /* ================================ BARQ ================================
               Third input path, and the only one that is not delivered to us: BARQ
               is a single-writer double-buffer in shared memory with no notification
               mechanism (the futex wake is still a TODO in its README), so the frame
               arrives when a timer goes looking for it.

               Wire layout, fixed at compile time on both ends:

                 struct BARQFrameHeader { uint32 width; uint32 height;
                                          uint32 point_step; double timestamp; }  // 20 B
                 struct BARQPoint       { float x, y, z, intensity;
                                          uint16 ring; double timestamp; }        // 26 B

               Byte-for-byte the same per-point layout as mmr_base/BoundedPointcloud,
               so fromBARQtoLimo below is fromROStoLimo(BoundedPointcloud) with a
               different pointer source -- see the offsets in both.               */

            void loadBarqConfig(const std::string& fallback_topic){
                /* Optional keys, read the same guarded way as the rest of this
                   node's config: parameters exist only if the YAML supplies them
                   (automatically_declare_parameters_from_overrides), and
                   declare_parameter here would throw AlreadyDeclared. */
                auto get_str = [&](const char* k, const std::string& d){
                    return this->has_parameter(k) ? this->get_parameter(k).as_string() : d; };
                auto get_int = [&](const char* k, int d){
                    return this->has_parameter(k) ? static_cast<int>(this->get_parameter(k).as_int()) : d; };
                auto get_dbl = [&](const char* k, double d){
                    return this->has_parameter(k) ? this->get_parameter(k).as_double() : d; };

                /* Defaults to the ROS topic name, because the Hesai driver names the
                   shared-memory segment after it (BARQ_topic in its config) and the
                   two are the same string on every stack we run. */
                this->barq_topic_             = get_str("BARQ_topic", fallback_topic);
                this->barq_retry_delay_ms_    = get_int("BARQ_retry_delay_ms", 100);
                /* 100 x 100 ms = 10 s, and it is deliberately generous: the stack is
                   started downstream-first and the driver -- the only process that
                   creates the segment -- comes up seconds later. cuda_cone_rush had
                   this budget too short and spent a whole study on the ROS2 fallback
                   while labelling every row transport=barq. This node has no such
                   fallback (see barqTryConnect), but the wait still has to cover the
                   warmup or it dies on a race instead of running. */
                this->barq_max_retries_       = get_int("BARQ_max_retries", 100);
                this->barq_polling_rate_ms_   = get_dbl("BARQ_polling_rate_ms", 1.0);
                this->barq_writer_timeout_ms_ = get_int("BARQ_writer_timeout_ms", 1000);

                /* MUST match the writer's kMaxPoints. Reader::init() derives the
                   address of the second buffer from this value alone
                   (base + alignUp(max_size, 64)), so a mismatch does not fail --
                   it silently reads garbage on every other frame. */
                const size_t kMaxPoints = 300000;
                this->barq_max_size_ = sizeof(BARQFrameHeader) + kMaxPoints * sizeof(BARQPoint) + 512;
            }

            void barqReaderInit(){
                RCLCPP_INFO(this->get_logger(), "BARQ: attaching to segment '%s' (poll %.4f ms)",
                            this->barq_topic_.c_str(), this->barq_polling_rate_ms_);
                this->barq_retries_ = 0;
                /* A timer, not a retry loop: this runs from the constructor, and
                   blocking here would also delay the IMU subscription created just
                   below it. The IMU stream does not wait for the LiDAR -- burning
                   ten seconds of it before subscribing costs the calibration window
                   for no reason. */
                this->barq_connect_timer_ = create_wall_timer(
                        std::chrono::milliseconds(this->barq_retry_delay_ms_),
                        std::bind(&LimoWrapper::barqTryConnect, this), this->lidar_cbg_);
            }

            void barqTryConnect(){
                this->barq_reader_ = std::make_unique<BARQ::Reader>(this->barq_topic_, this->barq_max_size_);

                /* isWriterAlive as well as init, because a segment OUTLIVES the
                   process that made it: Writer::destroy() unlinks it, but a driver
                   that is killed rather than shut down never gets there and leaves
                   a perfectly mappable region behind with a stale heartbeat.
                   Attaching to that succeeds, delivers nothing, and is torn down a
                   second later by the health check -- observed as a re-attach loop
                   once per second until the stale segment finally went away.
                   Writer::init() stamps the heartbeat before the first write, so a
                   genuinely fresh writer passes this immediately. */
                if(this->barq_reader_->init() &&
                   this->barq_reader_->isWriterAlive(static_cast<uint32_t>(this->barq_writer_timeout_ms_))){
                    this->barq_connect_timer_->cancel();
                    this->barq_connect_timer_.reset();
                    this->barq_retries_ = 0;

                    /* duration<double, milli>, NOT std::chrono::milliseconds: the
                       latter has an integral rep, so its converting constructor is
                       disabled for a floating-point argument and a sub-millisecond
                       period does not truncate, it fails to compile. */
                    this->barq_poll_timer_ = create_wall_timer(
                            std::chrono::duration<double, std::milli>(this->barq_polling_rate_ms_),
                            std::bind(&LimoWrapper::barqPoll, this), this->lidar_cbg_);
                    this->barq_health_timer_ = create_wall_timer(
                            std::chrono::milliseconds(this->barq_writer_timeout_ms_),
                            std::bind(&LimoWrapper::barqCheckWriterHealth, this), this->lidar_cbg_);

                    RCLCPP_INFO(this->get_logger(), "BARQ: reader attached to '%s'", this->barq_topic_.c_str());
                    return;
                }

                this->barq_reader_.reset();
                if(++this->barq_retries_ >= this->barq_max_retries_){
                    /* NO ROS2 FALLBACK, on purpose, and this is the one place where
                       this node deliberately dies rather than degrade. A reader that
                       quietly switches transport keeps publishing perfectly good
                       odometry under a label that is now a lie, and the only symptom
                       is a latency number that is not the one anybody meant to take.
                       Losing the node is loud; losing the measurement is not. */
                    RCLCPP_ERROR_STREAM(this->get_logger(), "\n-------------------------------------------------------------------\n"
                                        << "FAST_LIMO::FATAL ERROR: no BARQ writer on '" << this->barq_topic_ << "' after "
                                        << this->barq_max_retries_ << " x " << this->barq_retry_delay_ms_ << " ms.\n"
                                        << "          Is the Hesai driver up with BARQ_enable: true and a matching\n"
                                        << "          BARQ_topic? There is no ROS2 fallback here by design -- it would\n"
                                        << "          silently turn a BARQ run into a ROS2 run.\n"
                                        << "-------------------------------------------------------------------\n"
                                        );
                    this->barq_connect_timer_->cancel();
                    rclcpp::shutdown();
                    return;
                }
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                        "BARQ: no writer on '%s' yet (%d/%d)", this->barq_topic_.c_str(),
                        this->barq_retries_, this->barq_max_retries_);
            }

            void barqCheckWriterHealth(){
                if(!this->barq_reader_ || this->barq_reader_->isWriterAlive(
                            static_cast<uint32_t>(this->barq_writer_timeout_ms_))) return;

                RCLCPP_ERROR(this->get_logger(), "BARQ: writer heartbeat lost on '%s', re-attaching",
                             this->barq_topic_.c_str());

                /* Tear the old timers down BEFORE re-attaching. barqReaderInit only
                   creates, so re-entering it while the poll timer still exists would
                   leave a second one running against a stale reader. */
                this->barq_poll_timer_->cancel();
                this->barq_poll_timer_.reset();
                this->barq_health_timer_->cancel();
                this->barq_health_timer_.reset();
                this->barq_reader_->destroy();
                this->barq_reader_.reset();

                this->barqReaderInit();
            }

            /* BARQ entry point. Same pipeline as the other two, different source. */
            void barqPoll(){
                size_t  sz = 0;
                int64_t ts = 0;
                const void* ptr = this->barq_reader_->getLatest(sz, ts);

                // getLatest() returns nullptr when the sequence number has not moved
                if(!ptr || sz == 0) return;

#ifdef LATENCY_TESTING
                /* The POLL THAT FOUND THE DATA, not the instant it arrived -- there
                   is no callback here to be woken. So this arm's delivery figure
                   carries up to BARQ_polling_rate_ms of poll (half of it on average)
                   that is not transport. Say so wherever it is plotted; it is the
                   one arm whose number is not purely the transport's. */
                this->latency_t_in_ = latencyMonotonicNs();
#endif

                fast_limo::Localizer& loc = fast_limo::Localizer::getInstance();
                static bool pc_in_good_shape = this->checkPointcloudStructure(loc.get_sensor_type());

                if(not pc_in_good_shape){
                    throw std::runtime_error("FAST_LIMO::FATAL ERROR: invalid pointcloud structure\n\n");
                }

                const uint8_t* buf = static_cast<const uint8_t*>(ptr);

                BARQFrameHeader hdr;
                if(sz < sizeof(hdr)){
                    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                            "BARQ: runt frame (%zu bytes, header is %zu)", sz, sizeof(hdr));
                    return;
                }
                std::memcpy(&hdr, buf, sizeof(hdr));

                /* Checked on EVERY frame, not behind a verbose flag: this is a raw
                   pointer into another process's memory and the length is the only
                   thing standing between a short write and a read past the buffer. */
                if(hdr.point_step != sizeof(BARQPoint) ||
                   sizeof(hdr) + static_cast<size_t>(hdr.width) * hdr.point_step > sz){
                    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                            "BARQ: bad frame (width %u, point_step %u, %zu bytes) -- dropped",
                            hdr.width, hdr.point_step, sz);
                    return;
                }

                pcl::PointCloud<PointType>::Ptr pc_ (std::make_shared<pcl::PointCloud<PointType>>());
                this->fromBARQtoLimo(buf + sizeof(hdr), hdr.width, *pc_);

                /* Frame START time, in seconds, and already on the same clock as the
                   PointCloud2 header stamp and the BoundedPointcloud stamp: the
                   driver adds driver_start_timestamp_ to it (and to every per-point
                   time, which is what the deskew below reads as max_point_time). */
                this->lidar_callback(pc_, hdr.timestamp);
            }

            void fromBARQtoLimo(const uint8_t* points, uint32_t n_points, pcl::PointCloud<PointType>& out){

                out.points.resize(n_points);
                out.width    = n_points;
                out.height   = 1;
                out.is_dense = false;

                const uint8_t* p = points;
                for(uint32_t i=0; i < n_points; i++, p += sizeof(BARQPoint)){
                    PointType& pt = out.points[i];
                    std::memcpy(&pt.x,         p +  0, 4);
                    std::memcpy(&pt.y,         p +  4, 4);
                    std::memcpy(&pt.z,         p +  8, 4);
                    std::memcpy(&pt.intensity, p + 12, 4);
                    std::memcpy(&pt.timestamp, p + 18, 8); // NOTE: ring (offset 16, UINT16) is unused by fast_limo
                }
            }
            #endif

            /* Common LiDAR pipeline (input agnostic). All outputs are standard PointCloud2. */
            void lidar_callback(pcl::PointCloud<PointType>::Ptr& pc_, double stamp) {

                fast_limo::Localizer& loc = fast_limo::Localizer::getInstance();

                loc.updatePointCloud(pc_, stamp);

#ifdef LATENCY_TESTING
                /* t_out HERE, before the publishes below, and this node is the
                   one place in the stack where that is not the usual rule. */
                const int64_t t_out = latencyMonotonicNs();
#endif

                /* NOTE: every pointcloud below is serialized on publish (~32 B/point), so all of
                    them are gated on having at least one subscriber. Nobody listening, no cost. */

                // Publish output pointcloud
                this->publishPointCloud(this->pc_pub, loc.get_pointcloud(), this->world_frame);

                // Publish debugging pointclouds
                if(this->debug){
                this->publishPointCloud(this->orig_pub,     loc.get_orig_pointcloud(),     this->lidar_frame);
                this->publishPointCloud(this->desk_pub,     loc.get_deskewed_pointcloud(), this->world_frame);
                this->publishPointCloud(this->match_pub,    loc.get_pc2match_pointcloud(), this->body_frame);
                this->publishPointCloud(this->finalraw_pub, loc.get_finalraw_pointcloud(), this->world_frame);

                // Visualize current matches
                if(this->match_points_pub->get_subscription_count() > 0){
                    visualization_msgs::msg::MarkerArray match_markers = this->getMatchesMarker(loc.get_matches(),
                                                                                            this->world_frame
                                                                                            );
                    this->match_points_pub->publish(match_markers);
                }
                }

#ifdef LATENCY_TESTING
                /* Published LAST, after both instants are already in hand, so the
                   instrumentation is outside every interval it reports.

                   `stamp` is the cloud's capture time in seconds, the same value
                   the driver's frame tick carries -- so this joins against the
                   tick with no per-transport special casing. A double at epoch
                   scale has ~240 ns of ULP, well inside the monitor's 1 ms
                   match window. */
                mmr_base::msg::LatencySample lm;
                lm.frame_stamp = rclcpp::Time(static_cast<int64_t>(stamp * 1e9));
                lm.t_in = this->latency_t_in_;
                lm.t_out = t_out;
                lm.seq = this->latency_seq_++;
                if (this->latency_sample_pub_) this->latency_sample_pub_->publish(lm);
#endif
            }

            void imu_callback(const sensor_msgs::msg::Imu & msg) {

                fast_limo::Localizer& loc = fast_limo::Localizer::getInstance();

                fast_limo::IMUmeas imu;
                this->fromROStoLimo(msg, imu);

                // Propagate IMU measurement
                loc.updateIMU(imu);

                // State publishing
                nav_msgs::msg::Odometry state_msg; 
                this->fromLimoToROS(loc.getWorldState(), loc.getPoseCovariance(), loc.getTwistCovariance(), state_msg);
                this->state_pub->publish(state_msg);
                
                if(this->debug){
                nav_msgs::msg::Odometry body_msg;
                this->fromLimoToROS(loc.getBodyState(), loc.getPoseCovariance(), loc.getTwistCovariance(), body_msg);
                this->body_pub->publish(body_msg);
                }

                // TF broadcasting
                if(this->publish_tf)
                    this->broadcastTF(loc.getWorldState(), world_frame, body_frame, true);
            }

        /* ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
           ///////////////////////////////////////             Load params          ///////////////////////////////////////////////////////////// 
           ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// */

           void loadConfig(fast_limo::Config* config){

                // Topics
                rclcpp::Parameter lidar_topic_p = this->get_parameter("topics.input.lidar");
                rclcpp::Parameter imu_topic_p   = this->get_parameter("topics.input.imu");
                config->topics.lidar = lidar_topic_p.as_string();
                config->topics.imu = imu_topic_p.as_string();

                // Frames
                rclcpp::Parameter world_p = this->get_parameter("frames.world");
                rclcpp::Parameter body_p = this->get_parameter("frames.body");
                this->world_frame = world_p.as_string();
                this->body_frame = body_p.as_string();

                /* The frame the raw input cloud arrives in. Only /fast_limo/original
                   is published in it -- that topic is the untransformed input scan,
                   so labelling it `body` claims an extrinsic that has not been
                   applied (see docs/FRAMES.md rule 8). Optional: params files
                   predating this key fall back to `body`, which is what the topic
                   used to claim. */
                rclcpp::Parameter lidar_p;
                if (this->get_parameter("frames.lidar", lidar_p))
                    this->lidar_frame = lidar_p.as_string();
                else
                    this->lidar_frame = this->body_frame;

                // General
                rclcpp::Parameter n_thread_p = this->get_parameter("num_threads");
                config->num_threads = n_thread_p.as_int();
                rclcpp::Parameter sensor_p = this->get_parameter("sensor_type");
                config->sensor_type = sensor_p.as_int();
                rclcpp::Parameter debug_p = this->get_parameter("debug");
                config->debug = debug_p.as_bool();
                rclcpp::Parameter verbose_p = this->get_parameter("verbose");
                config->verbose = verbose_p.as_bool();
                rclcpp::Parameter extr_p = this->get_parameter("estimate_extrinsics");
                config->ikfom.estimate_extrinsics = extr_p.as_bool();
                rclcpp::Parameter offset_p = this->get_parameter("time_offset");
                config->time_offset = offset_p.as_bool();
                /* Parameters here come from the YAML via
                   automatically_declare_parameters_from_overrides, so this must
                   NOT declare_parameter (that throws AlreadyDeclared). Guarded
                   so an older params file without the key still starts. */
                config->time_offset_tau = this->has_parameter("time_offset_tau")
                        ? this->get_parameter("time_offset_tau").as_double() : 0.0;
                rclcpp::Parameter eos_p = this->get_parameter("end_of_sweep");
                config->end_of_sweep = eos_p.as_bool();

                // Calibration
                rclcpp::Parameter grav_p = this->get_parameter("calibration.gravity_align");
                config->gravity_align = grav_p.as_bool();
                rclcpp::Parameter est_accel_p = this->get_parameter("calibration.accel");
                config->calibrate_accel = est_accel_p.as_bool();
                rclcpp::Parameter est_gyro_p = this->get_parameter("calibration.gyro");
                config->calibrate_gyro = est_gyro_p.as_bool();
                rclcpp::Parameter est_time_p = this->get_parameter("calibration.time");
                config->imu_calib_time = est_time_p.as_double();

                // Extrinsics
                rclcpp::Parameter extr_imu_t_p = this->get_parameter("extrinsics.imu.t");
                std::vector<double> imu2baselink_t = extr_imu_t_p.as_double_array();
                config->extrinsics.imu2baselink_t = std::vector<float>(imu2baselink_t.begin(), imu2baselink_t.end());
                rclcpp::Parameter extr_imu_R_p = this->get_parameter("extrinsics.imu.R");
                std::vector<double> imu2baselink_R = extr_imu_R_p.as_double_array();
                config->extrinsics.imu2baselink_R = std::vector<float>(imu2baselink_R.begin(), imu2baselink_R.end());
                rclcpp::Parameter extr_lidar_t_p = this->get_parameter("extrinsics.lidar.t");
                std::vector<double> lidar2baselink_t = extr_lidar_t_p.as_double_array();
                config->extrinsics.lidar2baselink_t = std::vector<float>(lidar2baselink_t.begin(), lidar2baselink_t.end());
                rclcpp::Parameter extr_lidar_R_p = this->get_parameter("extrinsics.lidar.R");
                std::vector<double> lidar2baselink_R = extr_lidar_R_p.as_double_array();
                config->extrinsics.lidar2baselink_R = std::vector<float>(lidar2baselink_R.begin(), lidar2baselink_R.end());

                // Intrinsics
                rclcpp::Parameter intr_accel_bias_p = this->get_parameter("intrinsics.accel.bias");
                std::vector<double> accel_bias = intr_accel_bias_p.as_double_array();
                config->intrinsics.accel_bias = std::vector<float>(accel_bias.begin(), accel_bias.end());
                rclcpp::Parameter intr_accel_sm_p = this->get_parameter("intrinsics.accel.sm");
                std::vector<double> imu_sm = intr_accel_sm_p.as_double_array();
                config->intrinsics.imu_sm = std::vector<float>(imu_sm.begin(), imu_sm.end());
                rclcpp::Parameter intr_gyro_bias_p = this->get_parameter("intrinsics.gyro.bias");
                std::vector<double> gyro_bias = intr_gyro_bias_p.as_double_array();
                config->intrinsics.gyro_bias = std::vector<float>(gyro_bias.begin(), gyro_bias.end());

                // Crop Box filter
                rclcpp::Parameter filt_crop_flag_p = this->get_parameter("filters.cropBox.active");
                config->filters.crop_active = filt_crop_flag_p.as_bool();
                rclcpp::Parameter filt_crop_min_p = this->get_parameter("filters.cropBox.box.min");
                std::vector<double> cropBoxMin = filt_crop_min_p.as_double_array();
                config->filters.cropBoxMin = std::vector<float>(cropBoxMin.begin(), cropBoxMin.end());
                rclcpp::Parameter filt_crop_max_p = this->get_parameter("filters.cropBox.box.max");
                std::vector<double> cropBoxMax = filt_crop_max_p.as_double_array();
                config->filters.cropBoxMax = std::vector<float>(cropBoxMax.begin(), cropBoxMax.end());

                // Voxel Grid filter
                rclcpp::Parameter filt_voxel_flag_p = this->get_parameter("filters.voxelGrid.active");
                config->filters.voxel_active = filt_voxel_flag_p.as_bool();
                rclcpp::Parameter filt_voxel_size_p = this->get_parameter("filters.voxelGrid.leafSize");
                std::vector<double> leafSize = filt_voxel_size_p.as_double_array();
                config->filters.leafSize = std::vector<float>(leafSize.begin(), leafSize.end());

                // Sphere crop filter
                rclcpp::Parameter filt_dist_flag_p = this->get_parameter("filters.minDistance.active");
                config->filters.dist_active = filt_dist_flag_p.as_bool();
                rclcpp::Parameter filt_dist_val_p = this->get_parameter("filters.minDistance.value");
                config->filters.min_dist = filt_dist_val_p.as_double();

                // FoV filter
                rclcpp::Parameter filt_fov_flag_p = this->get_parameter("filters.FoV.active");
                config->filters.fov_active = filt_fov_flag_p.as_bool();
                rclcpp::Parameter filt_fov_val_p = this->get_parameter("filters.FoV.value");
                config->filters.fov_angle = static_cast<float>(filt_fov_val_p.as_double()*M_PI/360.0); // half of FoV (bc. is divided by the x-axis)

                // Sampling Rate filter
                rclcpp::Parameter filt_rate_flag_p = this->get_parameter("filters.rateSampling.active");
                config->filters.rate_active = filt_rate_flag_p.as_bool();
                rclcpp::Parameter filt_rate_val_p = this->get_parameter("filters.rateSampling.value");
                config->filters.rate_value = filt_rate_val_p.as_int();

                // iKFoM config
                rclcpp::Parameter max_iters_p = this->get_parameter("iKFoM.MAX_NUM_ITERS");
                config->ikfom.MAX_NUM_ITERS = max_iters_p.as_int();
                rclcpp::Parameter max_match_p = this->get_parameter("iKFoM.MAX_NUM_MATCHES");
                config->ikfom.mapping.MAX_NUM_MATCHES = max_match_p.as_int();
                rclcpp::Parameter max_pc_p = this->get_parameter("iKFoM.MAX_NUM_PC2MATCH");
                config->ikfom.mapping.MAX_NUM_PC2MATCH = max_pc_p.as_int();
                rclcpp::Parameter limits_p = this->get_parameter("iKFoM.LIMITS");
                config->ikfom.LIMITS = std::vector<double>(23, limits_p.as_double());

                // Mapping
                rclcpp::Parameter match_point_p = this->get_parameter("iKFoM.Mapping.NUM_MATCH_POINTS");
                config->ikfom.mapping.NUM_MATCH_POINTS = match_point_p.as_int();
                rclcpp::Parameter max_plane_dist_p = this->get_parameter("iKFoM.Mapping.MAX_DIST_PLANE");
                config->ikfom.mapping.MAX_DIST_PLANE = max_plane_dist_p.as_double();
                rclcpp::Parameter plane_thr_p = this->get_parameter("iKFoM.Mapping.PLANES_THRESHOLD");
                config->ikfom.mapping.PLANE_THRESHOLD = plane_thr_p.as_double();

                // iOcTree
                rclcpp::Parameter bucket_size = this->get_parameter("iKFoM.Mapping.Octree.bucket_size");
                config->ikfom.mapping.octree.bucket_size = bucket_size.as_int();
                rclcpp::Parameter min_extent = this->get_parameter("iKFoM.Mapping.Octree.min_extent");
                config->ikfom.mapping.octree.min_extent = static_cast<float>(min_extent.as_double());
                rclcpp::Parameter downsampling = this->get_parameter("iKFoM.Mapping.Octree.downsampling");
                config->ikfom.mapping.octree.downsampling = downsampling.as_bool();

                /* Map backend selection + hash grid settings.
                   Read defensively (has_parameter) so a config predating the hashgrid
                   backend still launches on the octree exactly as it did before. */
                config->ikfom.mapping.backend = "octree";
                if(this->has_parameter("iKFoM.Mapping.backend"))
                    config->ikfom.mapping.backend = this->get_parameter("iKFoM.Mapping.backend").as_string();

                config->ikfom.mapping.hash_grid.voxel_size = 0.5f;
                if(this->has_parameter("iKFoM.Mapping.HashGrid.voxel_size"))
                    config->ikfom.mapping.hash_grid.voxel_size =
                        static_cast<float>(this->get_parameter("iKFoM.Mapping.HashGrid.voxel_size").as_double());

                config->ikfom.mapping.hash_grid.max_points_per_voxel = 20;
                if(this->has_parameter("iKFoM.Mapping.HashGrid.max_points_per_voxel"))
                    config->ikfom.mapping.hash_grid.max_points_per_voxel =
                        this->get_parameter("iKFoM.Mapping.HashGrid.max_points_per_voxel").as_int();

                config->ikfom.mapping.hash_grid.neighbors = 7;
                if(this->has_parameter("iKFoM.Mapping.HashGrid.neighbors"))
                    config->ikfom.mapping.hash_grid.neighbors =
                        this->get_parameter("iKFoM.Mapping.HashGrid.neighbors").as_int();

                // Covariance
                rclcpp::Parameter gyro_p = this->get_parameter("iKFoM.covariance.gyro");
                config->ikfom.cov_gyro = gyro_p.as_double();
                rclcpp::Parameter accel_p = this->get_parameter("iKFoM.covariance.accel");
                config->ikfom.cov_acc = accel_p.as_double();
                rclcpp::Parameter gyro_bias_p = this->get_parameter("iKFoM.covariance.bias_gyro");
                config->ikfom.cov_bias_gyro = gyro_bias_p.as_double();
                rclcpp::Parameter accel_bias_p = this->get_parameter("iKFoM.covariance.bias_accel");
                config->ikfom.cov_bias_acc = accel_bias_p.as_double();
           }

        
        /* ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
           ///////////////////////////////////////             Aux. func.           ///////////////////////////////////////////////////////////// 
           ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// */

            void fromROStoLimo(const sensor_msgs::msg::Imu& in, fast_limo::IMUmeas& out){
                out.stamp = rclcpp::Time(in.header.stamp).seconds();

                out.ang_vel(0) = in.angular_velocity.x;
                out.ang_vel(1) = in.angular_velocity.y;
                out.ang_vel(2) = in.angular_velocity.z;

                out.lin_accel(0) = in.linear_acceleration.x;
                out.lin_accel(1) = in.linear_acceleration.y;
                out.lin_accel(2) = in.linear_acceleration.z;

                Eigen::Quaterniond qd(in.orientation.w, 
                                    in.orientation.x, 
                                    in.orientation.y, 
                                    in.orientation.z );
                out.q = qd.cast<float>();
            }

            void publishPointCloud(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& pub,
                                    pcl::PointCloud<PointType>::ConstPtr pc, const std::string& frame_id){

                if(pub->get_subscription_count() < 1) return; // don't pay for the serialization if nobody is listening

                sensor_msgs::msg::PointCloud2 pc_ros;
                pcl::toROSMsg(*pc, pc_ros);
                pc_ros.header.stamp = this->get_clock()->now();
                pc_ros.header.frame_id = frame_id;
                pub->publish(pc_ros);
            }

            #ifdef ENABLE_ZERO_COPY
            void fromROStoLimo(const mmr_base::msg::BoundedPointcloud& in, pcl::PointCloud<PointType>& out){

                using BPC = mmr_base::msg::BoundedPointcloud;

                const uint32_t n_points = std::min(in.width, BPC::DATA_MAX_POINTS);

                out.points.resize(n_points);
                out.width    = n_points;
                out.height   = 1;
                out.is_dense = false;

                const uint8_t* p = in.data.data();
                for(uint32_t i=0; i < n_points; i++, p += BPC::POINT_STEP){
                    PointType& pt = out.points[i];
                    std::memcpy(&pt.x,         p +  0, 4);
                    std::memcpy(&pt.y,         p +  4, 4);
                    std::memcpy(&pt.z,         p +  8, 4);
                    std::memcpy(&pt.intensity, p + 12, 4);
                    std::memcpy(&pt.timestamp, p + 18, 8); // NOTE: ring (offset 16, UINT16) is unused by fast_limo
                }
            }
            #endif

            void fromLimoToROS(const fast_limo::State& in, nav_msgs::msg::Odometry& out){
                out.header.stamp = this->get_clock()->now();
                /* Both frames come from the "frames" params, the same ones
                   broadcastTF() uses, so the message and the TF cannot disagree.
                   frame_id used to be hardcoded "map" (wrong as soon as
                   frames.world is changed) and child_frame_id was never set at
                   all, which leaves it empty: malformed by the nav_msgs/Odometry
                   contract, and it propagates -- cuda_cone_fused inherits it for
                   /Odometry and then broadcasts "track -> ", a transform strict
                   consumers such as Foxglove drop outright. */
                out.header.frame_id = this->world_frame;
                out.child_frame_id  = this->body_frame;

                // Pose/Attitude
                Eigen::Vector3d pos = in.p.cast<double>();
                out.pose.pose.position.x = pos(0);
                out.pose.pose.position.y = pos(1);
                out.pose.pose.position.z = pos(2);

                Eigen::Quaterniond quat = in.q.cast<double>();
                out.pose.pose.orientation.x = quat.x();
                out.pose.pose.orientation.y = quat.y();
                out.pose.pose.orientation.z = quat.z();
                out.pose.pose.orientation.w = quat.w();

                // Twist
                Eigen::Vector3d lin_v = in.v.cast<double>();
                out.twist.twist.linear.x  = lin_v(0);
                out.twist.twist.linear.y  = lin_v(1);
                out.twist.twist.linear.z  = lin_v(2);

                Eigen::Vector3d ang_v = in.w.cast<double>();
                out.twist.twist.angular.x = ang_v(0);
                out.twist.twist.angular.y = ang_v(1);
                out.twist.twist.angular.z = ang_v(2);
            }

            void fromLimoToROS(const fast_limo::State& in, const std::vector<double>& cov_pose,
                                const std::vector<double>& cov_twist, nav_msgs::msg::Odometry& out){

                this->fromLimoToROS(in, out);

                // Covariances
                for(long unsigned int i=0; i<cov_pose.size(); i++){
                    out.pose.covariance[i]  = cov_pose[i];
                    out.twist.covariance[i] = cov_twist[i];
                }
            }

            void broadcastTF(const fast_limo::State& in, std::string parent_name, std::string child_name, bool now){

                geometry_msgs::msg::TransformStamped tf_msg;
                tf_msg.header.stamp    = (now) ? this->get_clock()->now() : rclcpp::Time(in.time);
                /* NOTE: depending on IMU sensor rate, the state's stamp could be too old, 
                    so a TF warning could be print out (really annoying!).
                    In order to avoid this, the "now" argument should be true.
                */
                tf_msg.header.frame_id = parent_name;
                tf_msg.child_frame_id  = child_name;

                // Translation
                Eigen::Vector3d pos = in.p.cast<double>();
                tf_msg.transform.translation.x = pos(0);
                tf_msg.transform.translation.y = pos(1);
                tf_msg.transform.translation.z = pos(2);

                // Rotation
                Eigen::Quaterniond quat = in.q.cast<double>();
                tf_msg.transform.rotation.x = quat.x();
                tf_msg.transform.rotation.y = quat.y();
                tf_msg.transform.rotation.z = quat.z();
                tf_msg.transform.rotation.w = quat.w();

                // Broadcast
                tf_broadcaster_->sendTransform(tf_msg);
            }

            /* Shared by BOTH fixed-layout transports: BoundedPointcloud and the BARQ
               frame are the same 26 bytes per point (compare the two layouts), so
               the compatibility question they raise is one question, not two. */
            #if defined(ENABLE_ZERO_COPY) || defined(ENABLE_BARQ)
            bool checkPointcloudStructure(fast_limo::SensorType sensor){

                /* NOTE: neither BoundedPointcloud nor a BARQ frame carries PointField
                    metadata, their layout is fixed at compile time and the timestamp
                    field is an absolute FLOAT64, so they can only feed a HESAI/LIVOX
                    alike configuration. */

                if( (sensor == fast_limo::SensorType::HESAI) || (sensor == fast_limo::SensorType::LIVOX) )
                    return true;

                RCLCPP_ERROR_STREAM(this->get_logger(), "\n-------------------------------------------------------------------\n"
                                    << "FAST_LIMO::FATAL ERROR: fixed-layout input (BoundedPointcloud / BARQ) is only compatible\n"
                                    << "          with HESAI/LIVOX alike pointclouds, as its fixed layout is:\n"
                                    << "                  x: FLOAT32 (offset 0)\n"
                                    << "                  y: FLOAT32 (offset 4)\n"
                                    << "                  z: FLOAT32 (offset 8)\n"
                                    << "                  intensity: FLOAT32 (offset 12)\n"
                                    << "                  ring: UINT16 (offset 16)\n"
                                    << "                  timestamp: FLOAT64 (offset 18, global time in seconds)\n"
                                    << "          Either set sensor_type to 2 (HESAI) / 3 (LIVOX) or disable zero_copy/BARQ_enabled.\n"
                                    << "-------------------------------------------------------------------\n"
                                    );

                return false;
            }
            #endif

            bool checkPointcloudStructure(const sensor_msgs::msg::PointCloud2 & msg, fast_limo::SensorType sensor){

                using sensor_msgs::msg::PointField;

                if (sensor == fast_limo::SensorType::OUSTER) {
                    for(size_t i=0; i < msg.fields.size(); i++){
                        if( (msg.fields[i].name == "t") && (msg.fields[i].datatype == PointField::UINT32) )
                            return true;
                    }
            
                    RCLCPP_ERROR_STREAM(this->get_logger(), "\n-------------------------------------------------------------------\n" 
                                        << "FAST_LIMO::FATAL ERROR: the received pointcloud MUST have a timestamp field available!\n"
                                        << "          Remember that for OUSTER alike pointclouds, the expected fields are:\n"
                                        << "                  x: FLOAT32 (x coordinate in meters)\n"
                                        << "                  y: FLOAT32 (y coordinate in meters)\n"
                                        << "                  z: FLOAT32 (z coordinate in meters)\n"
                                        << "                  t: UINT32 (time since beginning of scan in nanoseconds)\n"
                                        << "-------------------------------------------------------------------\n"
                                        );
            
                } else if (sensor == fast_limo::SensorType::VELODYNE) {
                    for(size_t i=0; i < msg.fields.size(); i++){
                        if( (msg.fields[i].name == "time") && (msg.fields[i].datatype == PointField::FLOAT32)  )
                            return true;
                    }

                    RCLCPP_ERROR_STREAM(this->get_logger(), "\n-------------------------------------------------------------------\n" 
                                        << "FAST_LIMO::FATAL ERROR: the received pointcloud MUST have a timestamp field available!\n"
                                        << "          Remember that for VELODYNE alike pointclouds, the expected fields are:\n"
                                        << "                  x: FLOAT32 (x coordinate in meters)\n"
                                        << "                  y: FLOAT32 (y coordinate in meters)\n"
                                        << "                  z: FLOAT32 (z coordinate in meters)\n"
                                        << "                  time: FLOAT32 (time since beginning of scan in seconds)\n"
                                        << "-------------------------------------------------------------------\n"
                                        );
            
                } else if ( (sensor == fast_limo::SensorType::HESAI) || (sensor == fast_limo::SensorType::LIVOX) ) {
                    for(size_t i=0; i < msg.fields.size(); i++){
                        if( (msg.fields[i].name == "timestamp") && (msg.fields[i].datatype == PointField::FLOAT64) )
                            return true;
                    }

                    RCLCPP_ERROR_STREAM(this->get_logger(), "\n-------------------------------------------------------------------\n" 
                                        << "FAST_LIMO::FATAL ERROR: the received pointcloud MUST have a timestamp field available!\n"
                                        << "          Remember that for HESAI/LIVOX alike pointclouds, the expected fields are:\n"
                                        << "                  x: FLOAT32 (x coordinate in meters)\n"
                                        << "                  y: FLOAT32 (y coordinate in meters)\n"
                                        << "                  z: FLOAT32 (z coordinate in meters)\n"
                                        << "                  timestamp: FLOAT64 (global time in seconds/nanoseconds if HESAI/LIVOX)\n"
                                        << "-------------------------------------------------------------------\n"
                                        );
            
                } else {
                    RCLCPP_ERROR_STREAM(this->get_logger(), "\n-------------------------------------------------------------------\n" 
                                        << "FAST_LIMO::FATAL ERROR: LiDAR sensor type unknown or not specified!\n"
                                        << "-------------------------------------------------------------------\n"
                                        );
                }
                
                return false;
            }

            visualization_msgs::msg::MarkerArray getMatchesMarker(Matches& matches, std::string frame_id){
                visualization_msgs::msg::MarkerArray m_array;
                visualization_msgs::msg::Marker m;

                m_array.markers.reserve(matches.size());

                m.ns = "fast_limo_match";
                m.type = visualization_msgs::msg::Marker::SPHERE;
                m.action = visualization_msgs::msg::Marker::ADD;

                m.color.r = 0.0f;
                m.color.g = 0.0f;
                m.color.b = 1.0f;
                m.color.a = 1.0f;

                m.lifetime = rclcpp::Duration::from_seconds(0.0);
                m.header.frame_id = frame_id;
                m.header.stamp = this->get_clock()->now();

                m.pose.orientation.w = 1.0;

                m.scale.x = 0.2;
                m.scale.y = 0.2;
                m.scale.z = 0.2;

                for(size_t i=0; i < matches.size(); i++){
                    m.id = static_cast<int>(i);
                    Eigen::Vector3f match_p = matches[i].get_global_point();
                    m.pose.position.x = match_p(0);
                    m.pose.position.y = match_p(1);
                    m.pose.position.z = match_p(2);

                    m_array.markers.push_back(m);
                }

                return m_array;
            }

    };

}

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);

    rclcpp::Node::SharedPtr limo = std::make_shared<ros2wrap::LimoWrapper>();

    // Two threads, not one per core: the node has exactly two MutuallyExclusive
    // callback groups (LiDAR and IMU), so at most two callbacks can ever run at
    // once. Any extra executor thread can never pick up work -- it only wakes,
    // contends for the executor mutex and migrates, stealing cores from the
    // OpenMP regions inside the LiDAR callback.
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
    executor.add_node(limo);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}