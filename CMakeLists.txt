cmake_minimum_required(VERSION 3.5)
project(fast_limo)

# Default to C++17
add_compile_options(-std=c++17)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # -Wpedantic only ever fires inside the vendored IKFoM (mtk) headers, and
  # -Wpsabi is an aarch64 ABI note we cannot act on. IKFoM's own warnings are
  # muted by exposing it as a SYSTEM include dir (see the ikfom target below).
  add_compile_options(-Wall -Wextra -Wno-psabi)
endif()

# Silences boost/bind.hpp's deprecation pragma, pulled in by esekfom.hpp.
add_definitions(-DBOOST_BIND_GLOBAL_PLACEHOLDERS)

set(CMAKE_BUILD_TYPE "Release")

find_package( PCL 1.12 REQUIRED )
include_directories(SYSTEM ${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})
link_directories(${PCL_LIBRARY_DIRS})

find_package( Eigen3 REQUIRED )
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

include(FindOpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else(OPENMP_FOUND)
  message("ERROR: OpenMP could not be found.")
endif(OPENMP_FOUND)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(pcl_conversions REQUIRED)

#=======================================
# Zero Copy support
#=======================================
# Per-frame latency instrumentation: one mmr_base/LatencySample per processed
# sweep on /latency/sample/fast_limo. Needs mmr_base for exactly the same reason
# ENABLE_ZERO_COPY does, which is why it is env-var driven in the same way --
# `LATENCY_TESTING=1 colcon build` instruments the whole stack at once.
option(LATENCY_TESTING "Publish per-frame latency samples on /latency/sample/<key>" OFF)
if($ENV{LATENCY_TESTING})
  set(LATENCY_TESTING ON)
endif()
if(LATENCY_TESTING)
  find_package(mmr_base QUIET)
  if(NOT mmr_base_FOUND)
    message(FATAL_ERROR
      "LATENCY_TESTING requires mmr_base (for msg/LatencySample) and it was not found")
  endif()
  message(STATUS "Latency testing: ENABLED -- samples on /latency/sample/fast_limo")
  add_definitions(-DLATENCY_TESTING)
else()
  message(STATUS "Latency testing: DISABLED")
endif()

option(ENABLE_ZERO_COPY "Enable Zero Copy for ROS2 low latency comunication" OFF)

if($ENV{RMW_FASTRTPS_USE_QOS_FROM_XML})
  set(ENABLE_ZERO_COPY ON)
endif()

if(ENABLE_ZERO_COPY)
  find_package(mmr_base QUIET)
  if(mmr_base_FOUND)
    message(
      "============================================================\n"
      "-- ZERO COPY is ENABLED\n"
      "============================================================\n"
    )
    add_definitions(-DENABLE_ZERO_COPY)
  else()
    message(
      "============================================================\n"
      "-- ZERO COPY is DISABLED - mmr_base was NOT FOUND\n"
      "============================================================\n"
    )
  endif()
else()
  message(
    "============================================================\n"
    "-- ZERO COPY is DISABLED - PLEASE SET ENV RMW_FASTRTPS_USE_QOS_FROM_XML=1 BEFORE BUILDING TO USE ZERO COPY\n"
    "============================================================\n"
  )
endif()

#=======================================
# BARQ support
#=======================================
# Third transport, alongside plain ROS2 and Fast DDS zero copy. Env-var driven
# in the same way as ENABLE_ZERO_COPY and LATENCY_TESTING so one
# `ENABLE_BARQ=1 colcon build` instruments the whole stack at once, and matching
# how cuda_cone_rush -- the other BARQ reader on the same segment -- is built.
option(ENABLE_BARQ "Enable BARQ shared-memory transport for the input pointcloud" OFF)
if($ENV{ENABLE_BARQ})
  set(ENABLE_BARQ ON)
endif()

if(ENABLE_BARQ)
  find_package(barq QUIET)
  if(NOT barq_FOUND)
    # FATAL, unlike the zero copy block above, which quietly degrades when
    # mmr_base is missing. Asking for BARQ and silently getting a ROS2 build is
    # the exact failure that made a whole transport A/B measure the wrong arm.
    message(FATAL_ERROR
      "============================================================\n"
      "ENABLE_BARQ is ON, but the BARQ package was not found.\n"
      "This package is not available on rosdep or apt.\n"
      "You must clone and build it manually from:\n"
      "    cd ~/ros2_ws/src\n"
      "    git clone https://github.com/leonardonels/barq.git\n"
      "    cd ~/ros2_ws && colcon build\n"
      "Then rebuild this package with:\n"
      "    cd ~/ros2_ws && ENABLE_BARQ=1 colcon build --packages-up-to fast_limo --cmake-args -DCMAKE_BUILD_TYPE=Release\n"
      "============================================================\n"
    )
  endif()
  message(
    "============================================================\n"
    "-- BARQ is ENABLED\n"
    "============================================================\n"
  )
  add_definitions(-DENABLE_BARQ)
else()
  message(
    "============================================================\n"
    "-- BARQ is DISABLED - PLEASE SET ENV ENABLE_BARQ=1 BEFORE BUILDING TO USE BARQ\n"
    "============================================================\n"
  )
endif()

include_directories(
  include
  ${EIGEN3_INLCUDE_DIRS}
  ${PCL_INLCUDE_DIRS}
)

# Not all machines have <cpuid.h> available
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
file(WRITE ${CMAKE_BINARY_DIR}/test_cpuid.cpp "#include <cpuid.h>")
try_compile(HAS_CPUID ${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR}/test_cpuid.cpp)
file(REMOVE ${CMAKE_BINARY_DIR}/test_cpuid.cpp)
if(HAS_CPUID)
  add_compile_definitions(HAS_CPUID)
endif()

message("PCL version: ${PCL_VERSION}")

# IKFOM
add_library(ikfom
            include/IKFoM/IKFoM_toolkit
            include/IKFoM/use-ikfom.cpp
            )

target_link_libraries(ikfom ${PCL_LIBRARIES} ${OpenMP_LIBS})
# SYSTEM: IKFoM is vendored third-party code. Consumers reach it via
# `#include <use-ikfom.hpp>`, which resolves through -isystem, so GCC mutes
# warnings there and in every header it pulls in transitively.
target_include_directories(ikfom SYSTEM
  PUBLIC include/IKFoM/ include/IKFoM/IKFoM_toolkit
)

# Fast LIMO
add_library(${PROJECT_NAME} 
            include/fast_limo/Modules/Localizer.cpp
            include/fast_limo/Modules/Mapper.cpp

            include/fast_limo/Objects/State.cpp
            include/fast_limo/Objects/Plane.cpp
            include/fast_limo/Objects/Match.cpp
            include/fast_limo/Objects/Octree.hpp
            )
target_compile_options(${PROJECT_NAME} PRIVATE ${OpenMP_FLAGS})
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} ${OpenMP_LIBS} ikfom)

# ROS execs
add_executable(${PROJECT_NAME}_multi_exec
              src/LimoWrapper.cpp
              )
set(NODE_DEPENDENCIES
  rclcpp sensor_msgs geometry_msgs visualization_msgs nav_msgs tf2 tf2_ros pcl_conversions
)

if(mmr_base_FOUND)
  list(APPEND NODE_DEPENDENCIES "mmr_base")
endif()

if(barq_FOUND)
  list(APPEND NODE_DEPENDENCIES "barq")
endif()

ament_target_dependencies(${PROJECT_NAME}_multi_exec ${NODE_DEPENDENCIES})
target_link_libraries(${PROJECT_NAME}_multi_exec ${PROJECT_NAME} ${PCL_LIBRARIES})

# Binaries
install( TARGETS 
  ${PROJECT_NAME}_multi_exec
  DESTINATION lib/${PROJECT_NAME} )

# Install launch files.
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()