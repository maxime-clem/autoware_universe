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

#ifndef AUTOWARE__MPPI_OPTIMIZER__FIRST_ORDER_DUBINS_MPPI_COST_PARAMS_HPP_
#define AUTOWARE__MPPI_OPTIMIZER__FIRST_ORDER_DUBINS_MPPI_COST_PARAMS_HPP_

namespace autoware::mppi_optimizer
{

/** Host-side MPPI tunables; defaults match mppi_optimizer.param.yaml. */
struct FirstOrderDubinsMppiCostParams
{
  /** Softmax temperature for trajectory weighting (higher = softer weighting). */
  float lambda{1500.0F};
  float speed_coeff{500.0F};
  float track_coeff{1000.0F};
  float track_terminal_scale{10.0F};
  float heading_coeff{500.0F};
  float lateral_distance_coeff{0.0F};
  float lateral_yaw_error_coeff{0.0F};
  float track_center_coeff{0.0F};
  float corner_buffer_coeff{0.0F};
  float corner_safe_margin{0.3F};
  float boundary_threshold{1.5F};
  /** Distance inside boundary_threshold at which the gradual lateral barrier activates. */
  float lateral_boundary_soft_margin{0.2F};
  float lateral_boundary_barrier_weight{};
  float accel_cmd_coeff{0.0F};
  float steer_cmd_coeff{0.0F};
  float steer_rate_coeff{0.0F};
  /** Roughness penalty used when smoothing the cold-start reference before curvature fitting. */
  float nominal_spline_smoothing_weight{10.0F};
  float lateral_acceleration_coeff{300.0F};
  float lateral_jerk_coeff{300.0F};
  float longitudinal_jerk_coeff{10.0F};
  float obstacle_collision_margin{0.5F};
  float road_border_collision_margin{0.3F};
  float obstacle_safe_margin{0.5F};
  float obstacle_barrier_weight{};
  float road_border_safe_margin{0.3F};
  float road_border_barrier_weight{};
  float drivable_area_safe_margin{0.0F};
  float drivable_area_barrier_weight{2000.0F};
  float max_crash_penalty{100000.0F};
  float goal_pos_coeff{1000.0F};
  float goal_speed_coeff{0.0F};
  float goal_yaw_coeff{500.0F};
  float goal_terminal_scale{10.0F};
};

}  // namespace autoware::mppi_optimizer

#endif  // AUTOWARE__MPPI_OPTIMIZER__FIRST_ORDER_DUBINS_MPPI_COST_PARAMS_HPP_
