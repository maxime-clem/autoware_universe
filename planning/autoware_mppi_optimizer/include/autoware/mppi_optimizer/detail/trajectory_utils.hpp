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

#ifndef AUTOWARE__MPPI_OPTIMIZER__DETAIL__TRAJECTORY_UTILS_HPP_
#define AUTOWARE__MPPI_OPTIMIZER__DETAIL__TRAJECTORY_UTILS_HPP_

#include "autoware/mppi_optimizer/first_order_dubins_mppi_interface.hpp"

#include <autoware_vehicle_msgs/msg/steering_report.hpp>
#include <geometry_msgs/msg/accel_with_covariance_stamped.hpp>

#include <cstddef>
#include <optional>
#include <vector>

namespace autoware::mppi_optimizer::detail
{

inline constexpr int kMppiHorizon = 80;
inline constexpr float kMppiDt = 0.1F;

struct InitialState
{
  float x{0.0F};
  float y{0.0F};
  float yaw{0.0F};
  float velocity{0.0F};
  float acceleration{0.0F};
  float steering{0.0F};
};

struct ReferenceSample
{
  float time{0.0F};
  float x{0.0F};
  float y{0.0F};
  float yaw{0.0F};
  float velocity{0.0F};
};

struct OptimizedState
{
  float x{0.0F};
  float y{0.0F};
  float yaw{0.0F};
  float velocity{0.0F};
  float steering{0.0F};
};

[[nodiscard]] bool isOptimizationRequired(const Trajectory & trajectory);

void setInitialEngageVelocity(Trajectory & trajectory);

[[nodiscard]] InitialState makeInitialState(
  const Odometry & odometry,
  const std::optional<geometry_msgs::msg::AccelWithCovarianceStamped> & acceleration,
  const std::optional<autoware_vehicle_msgs::msg::SteeringReport> & steering_status,
  const FirstOrderDubinsMppiVehicleParams & vehicle_params);

[[nodiscard]] std::vector<ReferenceSample> buildReferenceHorizon(
  const Trajectory & trajectory, const InitialState & ego, const int horizon = kMppiHorizon,
  const float dt = kMppiDt, const size_t start_idx = 0U);

[[nodiscard]] std::vector<FirstOrderDubinsMppiControl> buildDiffusionNominalControl(
  const Trajectory & reference, std::size_t start_idx,
  const FirstOrderDubinsMppiVehicleParams & vehicle_params, int horizon = kMppiHorizon,
  float smoothing_weight = 10.0F);

[[nodiscard]] std::vector<FirstOrderDubinsMppiControl> buildForcedNominalControl(
  const std::vector<float> & acceleration_commands, const std::vector<float> & steering_commands,
  const FirstOrderDubinsMppiVehicleParams & vehicle_params, int horizon = kMppiHorizon);

[[nodiscard]] std::vector<FirstOrderDubinsMppiControl> shiftNominalControl(
  const std::vector<FirstOrderDubinsMppiControl> & previous, int horizon = kMppiHorizon);

[[nodiscard]] Trajectory buildOptimizedTrajectory(
  const Trajectory & input, const std::vector<OptimizedState> & post_step_states,
  const std::vector<FirstOrderDubinsMppiControl> & controls);

[[nodiscard]] std::vector<float> computeSmoothedSplineCurvature(
  const std::vector<double> & x, const std::vector<double> & y, float smoothing_weight);

[[nodiscard]] float clampSteeringToReachableRange(
  float desired_steering, float current_steering, float max_steer_rate, float dt, float max_steer);

}  // namespace autoware::mppi_optimizer::detail

#endif  // AUTOWARE__MPPI_OPTIMIZER__DETAIL__TRAJECTORY_UTILS_HPP_
