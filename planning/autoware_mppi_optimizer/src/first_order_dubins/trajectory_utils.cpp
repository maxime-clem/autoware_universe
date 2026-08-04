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

#include "autoware/mppi_optimizer/detail/trajectory_utils.hpp"

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/utils.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace autoware::mppi_optimizer::detail
{

namespace
{

geometry_msgs::msg::Quaternion quaternionFromYaw(const float yaw)
{
  tf2::Quaternion quaternion;
  quaternion.setRPY(0.0, 0.0, yaw);
  return tf2::toMsg(quaternion);
}

}  // namespace

bool isOptimizationRequired(const Trajectory & trajectory)
{
  const bool is_stopping = std::any_of(
    trajectory.points.begin(), trajectory.points.end(),
    [](const auto & point) { return point.longitudinal_velocity_mps < 0.02F; });

  double length = 0.0;
  for (std::size_t i = 0; i + 1U < trajectory.points.size(); ++i) {
    length += std::hypot(
      trajectory.points[i].pose.position.x - trajectory.points[i + 1U].pose.position.x,
      trajectory.points[i].pose.position.y - trajectory.points[i + 1U].pose.position.y);
  }
  const bool is_short = length < 4.0;
  return !is_stopping || !is_short;
}

void setInitialEngageVelocity(Trajectory & trajectory)
{
  constexpr float engage_velocity = 0.25F;
  if (trajectory.points.size() < 3U) {
    return;
  }
  const bool wants_to_move = std::any_of(
    trajectory.points.begin(), trajectory.points.end(),
    [](const auto & point) { return point.longitudinal_velocity_mps > engage_velocity; });
  if (wants_to_move && trajectory.points[0].longitudinal_velocity_mps < 0.05F) {
    trajectory.points[0].longitudinal_velocity_mps = engage_velocity;
    trajectory.points[1].longitudinal_velocity_mps = engage_velocity;
  }
}

InitialState makeInitialState(
  const Odometry & odometry,
  const std::optional<geometry_msgs::msg::AccelWithCovarianceStamped> & acceleration,
  const std::optional<autoware_vehicle_msgs::msg::SteeringReport> & steering_status,
  const FirstOrderDubinsMppiVehicleParams & vehicle_params)
{
  InitialState state;
  state.x = static_cast<float>(odometry.pose.pose.position.x);
  state.y = static_cast<float>(odometry.pose.pose.position.y);
  state.yaw = static_cast<float>(tf2::getYaw(odometry.pose.pose.orientation));
  state.velocity = static_cast<float>(odometry.twist.twist.linear.x);
  const float acceleration_value =
    acceleration.has_value() ? static_cast<float>(acceleration->accel.accel.linear.x) : 0.0F;
  state.acceleration =
    std::clamp(acceleration_value, vehicle_params.min_accel(), vehicle_params.max_accel());
  const float steering_value =
    steering_status.has_value() ? steering_status->steering_tire_angle : 0.0F;
  state.steering =
    std::clamp(steering_value, -vehicle_params.max_steer_angle, vehicle_params.max_steer_angle);
  return state;
}

std::vector<ReferenceSample> buildReferenceHorizon(
  const Trajectory & trajectory, const InitialState & ego, const int horizon, const float dt, const size_t start_idx)
{
  const size_t sample_count = std::max(0, horizon);
  std::vector<ReferenceSample> reference(static_cast<std::size_t>(sample_count));
  for (auto k = 0U; k < sample_count; ++k) {
    const size_t idx = std::min(k + start_idx, sample_count - 1);
    auto & sample = reference[static_cast<std::size_t>(idx)];
    sample.time = static_cast<float>(k + 1) * dt;
    if (trajectory.points.empty()) {
      sample.x = ego.x;
      sample.y = ego.y;
      sample.yaw = ego.yaw;
      sample.velocity = ego.velocity;
      continue;
    }

    const std::size_t index = std::min(static_cast<std::size_t>(k), trajectory.points.size() - 1U);
    const auto & point = trajectory.points[index];
    sample.x = static_cast<float>(point.pose.position.x);
    sample.y = static_cast<float>(point.pose.position.y);
    sample.yaw = static_cast<float>(tf2::getYaw(point.pose.orientation));
    sample.velocity = point.longitudinal_velocity_mps;
  }
  return reference;
}

std::vector<FirstOrderDubinsMppiControl> buildDiffusionNominalControl(
  const Trajectory & reference, const std::size_t start_idx,
  const FirstOrderDubinsMppiVehicleParams & vehicle_params, const int horizon)
{
  const int control_count = std::max(0, horizon);
  std::vector<FirstOrderDubinsMppiControl> nominal(static_cast<std::size_t>(control_count));
  if (reference.points.empty()) {
    return nominal;
  }

  for (int t = 0; t < control_count; ++t) {
    const std::size_t index =
      std::min(start_idx + static_cast<std::size_t>(t), reference.points.size() - 1U);
    const auto & point = reference.points[index];
    auto & control = nominal[static_cast<std::size_t>(t)];
    control.accel_cmd =
      std::clamp(point.acceleration_mps2, vehicle_params.min_accel(), vehicle_params.max_accel());

    float steering = point.front_wheel_angle_rad;
    if (std::abs(steering) <= 1.0E-6F && index + 1U < reference.points.size()) {
      const float minimum_chord_length = std::max(1.5F, vehicle_params.wheel_base * 0.5F);
      std::size_t lookahead_index = index + 1U;
      float chord_length = 0.0F;
      while (lookahead_index < reference.points.size()) {
        const auto & next = reference.points[lookahead_index];
        const float dx = static_cast<float>(next.pose.position.x - point.pose.position.x);
        const float dy = static_cast<float>(next.pose.position.y - point.pose.position.y);
        chord_length = std::hypot(dx, dy);
        if (chord_length >= minimum_chord_length) {
          break;
        }
        ++lookahead_index;
      }

      if (chord_length >= minimum_chord_length) {
        const auto & next = reference.points[lookahead_index];
        const float yaw0 = static_cast<float>(tf2::getYaw(point.pose.orientation));
        const float yaw1 = static_cast<float>(tf2::getYaw(next.pose.orientation));
        const float yaw_difference = std::atan2(std::sin(yaw1 - yaw0), std::cos(yaw1 - yaw0));
        steering = std::atan(vehicle_params.wheel_base * yaw_difference / chord_length);
      }
    }
    control.steer_cmd =
      std::clamp(steering, -vehicle_params.max_steer_angle, vehicle_params.max_steer_angle);
  }
  return nominal;
}

std::vector<FirstOrderDubinsMppiControl> buildForcedNominalControl(
  const std::vector<float> & acceleration_commands, const std::vector<float> & steering_commands,
  const FirstOrderDubinsMppiVehicleParams & vehicle_params, const int horizon)
{
  const int control_count = std::max(0, horizon);
  std::vector<FirstOrderDubinsMppiControl> nominal(static_cast<std::size_t>(control_count));
  for (int t = 0; t < control_count; ++t) {
    const auto index = static_cast<std::size_t>(t);
    const float acceleration =
      index < acceleration_commands.size()
        ? acceleration_commands[index]
        : (acceleration_commands.empty() ? 0.0F : acceleration_commands.back());
    const float steering = index < steering_commands.size()
                             ? steering_commands[index]
                             : (steering_commands.empty() ? 0.0F : steering_commands.back());
    nominal[index].accel_cmd =
      std::clamp(acceleration, vehicle_params.min_accel(), vehicle_params.max_accel());
    nominal[index].steer_cmd =
      std::clamp(steering, -vehicle_params.max_steer_angle, vehicle_params.max_steer_angle);
  }
  return nominal;
}

std::vector<FirstOrderDubinsMppiControl> shiftNominalControl(
  const std::vector<FirstOrderDubinsMppiControl> & previous, const int horizon)
{
  const int control_count = std::max(0, horizon);
  std::vector<FirstOrderDubinsMppiControl> nominal(static_cast<std::size_t>(control_count));
  if (previous.empty()) {
    return nominal;
  }
  for (int t = 0; t < control_count; ++t) {
    const std::size_t source = std::min(static_cast<std::size_t>(t) + 1U, previous.size() - 1U);
    nominal[static_cast<std::size_t>(t)] = previous[source];
  }
  return nominal;
}

Trajectory buildOptimizedTrajectory(
  const Trajectory & input, const std::vector<OptimizedState> & post_step_states,
  const std::vector<FirstOrderDubinsMppiControl> & controls)
{
  Trajectory output = input;
  const std::size_t optimized_count =
    std::min({output.points.size(), post_step_states.size(), controls.size()});
  for (std::size_t i = 0; i < optimized_count; ++i) {
    const auto & state = post_step_states[i];
    const auto & input_point = input.points[i];
    auto & output_point = output.points[i];
    output_point.pose.position.x = state.x;
    output_point.pose.position.y = state.y;
    output_point.pose.position.z = input_point.pose.position.z;
    output_point.pose.orientation = quaternionFromYaw(state.yaw);
    output_point.longitudinal_velocity_mps = state.velocity;
    output_point.acceleration_mps2 = controls[i].accel_cmd;
    output_point.front_wheel_angle_rad = state.steering;
  }
  return output;
}

}  // namespace autoware::mppi_optimizer::detail
