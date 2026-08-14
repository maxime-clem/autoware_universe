// Copyright 2026 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TEMPORAL_MPT__PATH_TRACKING_SOLVER_HPP_
#define TEMPORAL_MPT__PATH_TRACKING_SOLVER_HPP_

#include <cstddef>
#include <memory>
#include <vector>

namespace temporal_mpt
{

/** Time-indexed reference path (x, y, yaw, v) in world frame. */
struct PathTrackingReference
{
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> yaw;
  std::vector<double> v;
};

struct PathTrackingInitialState
{
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
  double v{0.0};
};

struct PathTrackingResult
{
  bool ok{false};
  int status{-1};
  /** Stage controls u[k] = (accel, steer_angle), length = horizon N. */
  std::vector<double> accel_cmd;
  std::vector<double> steer_cmd;
};

/**
 * Thin host wrapper around the temporal kinematic-bicycle acados OCP.
 * Public header intentionally avoids generated acados includes so other packages can link it.
 */
class PathTrackingSolver
{
public:
  PathTrackingSolver();
  ~PathTrackingSolver();

  PathTrackingSolver(const PathTrackingSolver &) = delete;
  PathTrackingSolver & operator=(const PathTrackingSolver &) = delete;
  PathTrackingSolver(PathTrackingSolver &&) noexcept;
  PathTrackingSolver & operator=(PathTrackingSolver &&) noexcept;

  /** Bicycle model parameters (CG to front / rear axle) [m]. */
  void setModelParameters(double lf, double lr);

  /**
   * Solve one path-tracking MPC step.
   * Uses the same ego-centered XY + yaw-bias reference setup as
   * TrajectoryTemporalMPTOptimizer::optimize_trajectory.
   */
  PathTrackingResult solve(
    const PathTrackingInitialState & x0, const PathTrackingReference & reference);

  /** Compiled OCP horizon length N (control stages). */
  static std::size_t horizon();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace temporal_mpt

#endif  // TEMPORAL_MPT__PATH_TRACKING_SOLVER_HPP_
