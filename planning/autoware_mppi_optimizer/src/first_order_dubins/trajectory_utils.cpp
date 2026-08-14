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

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <autoware/interpolation/spline_interpolation.hpp>

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

double evaluateQuadraticThroughPoints(
  const double query, const double s0, const double value0, const double s1, const double value1,
  const double s2, const double value2)
{
  const double basis0 = (query - s1) * (query - s2) / ((s0 - s1) * (s0 - s2));
  const double basis1 = (query - s0) * (query - s2) / ((s1 - s0) * (s1 - s2));
  const double basis2 = (query - s0) * (query - s1) / ((s2 - s0) * (s2 - s1));
  return basis0 * value0 + basis1 * value1 + basis2 * value2;
}

std::vector<float> smoothSecondDifference(
  const std::vector<float> & values, const float smoothing_weight)
{
  if (values.size() < 3U || smoothing_weight <= 0.0F) {
    return values;
  }
  if (!std::all_of(
        values.begin(), values.end(), [](const float value) { return std::isfinite(value); })) {
    return values;
  }

  const auto value_count = static_cast<Eigen::Index>(values.size());
  Eigen::MatrixXd second_difference = Eigen::MatrixXd::Zero(value_count - 2, value_count);
  for (Eigen::Index row = 0; row < value_count - 2; ++row) {
    second_difference(row, row) = 1.0;
    second_difference(row, row + 1) = -2.0;
    second_difference(row, row + 2) = 1.0;
  }
  const Eigen::MatrixXd system =
    Eigen::MatrixXd::Identity(value_count, value_count) +
    static_cast<double>(smoothing_weight) * second_difference.transpose() * second_difference;
  const Eigen::LDLT<Eigen::MatrixXd> solver(system);
  if (solver.info() != Eigen::Success) {
    return values;
  }

  Eigen::VectorXd raw_values(value_count);
  for (Eigen::Index i = 0; i < value_count; ++i) {
    raw_values[i] = static_cast<double>(values[static_cast<std::size_t>(i)]);
  }
  const Eigen::VectorXd smoothed_values = solver.solve(raw_values);
  if (solver.info() != Eigen::Success) {
    return values;
  }

  std::vector<float> result(values.size());
  for (Eigen::Index i = 0; i < value_count; ++i) {
    result[static_cast<std::size_t>(i)] = static_cast<float>(smoothed_values[i]);
  }
  return result;
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
  const Trajectory & trajectory, const InitialState & ego, const int horizon, const float dt,
  const size_t start_idx)
{
  const size_t sample_count = std::max(0, horizon);
  std::vector<ReferenceSample> reference(static_cast<std::size_t>(sample_count));

  if (trajectory.points.empty()) {
    for (auto & reference_sample : reference) {
      reference_sample.x = ego.x;
      reference_sample.y = ego.y;
      reference_sample.yaw = ego.yaw;
      reference_sample.velocity = ego.velocity;
    }
    return reference;
  }

  for (std::size_t k = 0; k < sample_count; ++k) {
    auto & sample = reference[k];
    sample.time = static_cast<float>(k + 1U) * dt;

    const std::size_t source_idx = std::min(k + start_idx, trajectory.points.size() - 1U);
    const auto & point = trajectory.points[source_idx];

    sample.x = static_cast<float>(point.pose.position.x);
    sample.y = static_cast<float>(point.pose.position.y);
    sample.yaw = static_cast<float>(tf2::getYaw(point.pose.orientation));
    sample.velocity = point.longitudinal_velocity_mps;
  }
  return reference;
}

std::vector<float> computeSmoothedSplineCurvature(
  const std::vector<double> & x, const std::vector<double> & y, const float smoothing_weight)
{
  std::vector<float> curvatures(x.size(), 0.0F);
  if (x.size() != y.size() || x.size() < 3U) {
    return curvatures;
  }
  if (!std::isfinite(x.front()) || !std::isfinite(y.front())) {
    return curvatures;
  }

  constexpr double kMinimumKnotDistance = 1.0E-6;
  std::vector<double> knots{0.0};
  std::vector<double> compact_x;
  std::vector<double> compact_y;
  compact_x.reserve(x.size());
  compact_y.reserve(y.size());
  compact_x.push_back(x.front());
  compact_y.push_back(y.front());
  std::vector<double> query_s(x.size(), 0.0);
  for (std::size_t i = 1U; i < x.size(); ++i) {
    if (!std::isfinite(x[i]) || !std::isfinite(y[i])) {
      return curvatures;
    }
    const double ds = std::hypot(x[i] - compact_x.back(), y[i] - compact_y.back());
    if (ds > kMinimumKnotDistance) {
      knots.push_back(knots.back() + ds);
      compact_x.push_back(x[i]);
      compact_y.push_back(y[i]);
    }
    query_s[i] = knots.back();
  }

  const auto point_count = static_cast<Eigen::Index>(knots.size());
  if (point_count < 3) {
    return curvatures;
  }

  Eigen::MatrixXd second_difference = Eigen::MatrixXd::Zero(point_count - 2, point_count);
  for (Eigen::Index row = 0; row < point_count - 2; ++row) {
    second_difference(row, row) = 1.0;
    second_difference(row, row + 1) = -2.0;
    second_difference(row, row + 2) = 1.0;
  }
  const double weight = std::max(0.0, static_cast<double>(smoothing_weight));
  const Eigen::MatrixXd system = Eigen::MatrixXd::Identity(point_count, point_count) +
                                 weight * second_difference.transpose() * second_difference;
  const Eigen::LDLT<Eigen::MatrixXd> solver(system);
  if (solver.info() != Eigen::Success) {
    return curvatures;
  }

  const Eigen::Map<const Eigen::VectorXd> raw_x(compact_x.data(), point_count);
  const Eigen::Map<const Eigen::VectorXd> raw_y(compact_y.data(), point_count);
  const Eigen::VectorXd smooth_x_vector = solver.solve(raw_x);
  const Eigen::VectorXd smooth_y_vector = solver.solve(raw_y);
  if (solver.info() != Eigen::Success) {
    return curvatures;
  }
  std::vector<double> smooth_x(smooth_x_vector.data(), smooth_x_vector.data() + point_count);
  std::vector<double> smooth_y(smooth_y_vector.data(), smooth_y_vector.data() + point_count);

  // Natural cubic splines impose zero second derivative at their endpoint knots. Pad the fitted
  // path using quadratic extrapolation so the real path endpoints are evaluated in the interior.
  const double first_step = knots[1] - knots[0];
  const double last_step = knots[knots.size() - 1U] - knots[knots.size() - 2U];
  const double before_s = knots.front() - first_step;
  const double after_s = knots.back() + last_step;
  const double before_x = evaluateQuadraticThroughPoints(
    before_s, knots[0], smooth_x[0], knots[1], smooth_x[1], knots[2], smooth_x[2]);
  const double before_y = evaluateQuadraticThroughPoints(
    before_s, knots[0], smooth_y[0], knots[1], smooth_y[1], knots[2], smooth_y[2]);
  const std::size_t last = smooth_x.size() - 1U;
  const double after_x = evaluateQuadraticThroughPoints(
    after_s, knots[last - 2U], smooth_x[last - 2U], knots[last - 1U], smooth_x[last - 1U],
    knots[last], smooth_x[last]);
  const double after_y = evaluateQuadraticThroughPoints(
    after_s, knots[last - 2U], smooth_y[last - 2U], knots[last - 1U], smooth_y[last - 1U],
    knots[last], smooth_y[last]);
  knots.insert(knots.begin(), before_s);
  knots.push_back(after_s);
  smooth_x.insert(smooth_x.begin(), before_x);
  smooth_x.push_back(after_x);
  smooth_y.insert(smooth_y.begin(), before_y);
  smooth_y.push_back(after_y);

  const autoware::interpolation::SplineInterpolation x_of_s(knots, smooth_x);
  const autoware::interpolation::SplineInterpolation y_of_s(knots, smooth_y);
  const auto dx_ds = x_of_s.getSplineInterpolatedDiffValues(query_s);
  const auto dy_ds = y_of_s.getSplineInterpolatedDiffValues(query_s);
  const auto d2x_ds2 = x_of_s.getSplineInterpolatedQuadDiffValues(query_s);
  const auto d2y_ds2 = y_of_s.getSplineInterpolatedQuadDiffValues(query_s);

  constexpr double kMinimumSpeedSquared = 1.0E-12;
  for (std::size_t i = 0U; i < x.size(); ++i) {
    const double speed_squared = dx_ds[i] * dx_ds[i] + dy_ds[i] * dy_ds[i];
    if (speed_squared <= kMinimumSpeedSquared) {
      continue;
    }
    curvatures[i] = static_cast<float>(
      (dx_ds[i] * d2y_ds2[i] - dy_ds[i] * d2x_ds2[i]) / (speed_squared * std::sqrt(speed_squared)));
  }
  return curvatures;
}

float clampSteeringToReachableRange(
  const float desired_steering, const float current_steering, const float max_steer_rate,
  const float dt, const float max_steer)
{
  const float absolute_limit = std::max(0.0F, max_steer);
  const float current = std::clamp(current_steering, -absolute_limit, absolute_limit);
  const float max_delta = std::max(0.0F, max_steer_rate) * std::max(0.0F, dt);
  const float lower = std::max(-absolute_limit, current - max_delta);
  const float upper = std::min(absolute_limit, current + max_delta);
  return std::clamp(desired_steering, lower, upper);
}

std::vector<FirstOrderDubinsMppiControl> buildDiffusionNominalControl(
  const Trajectory & reference, const std::size_t start_idx,
  const FirstOrderDubinsMppiVehicleParams & vehicle_params, const int horizon,
  const float smoothing_weight)
{
  const int control_count = std::max(0, horizon);
  std::vector<FirstOrderDubinsMppiControl> nominal(static_cast<std::size_t>(control_count));
  if (reference.points.empty()) {
    return nominal;
  }

  std::vector<double> reference_x;
  std::vector<double> reference_y;
  std::vector<float> reference_acceleration;
  reference_x.reserve(reference.points.size());
  reference_y.reserve(reference.points.size());
  reference_acceleration.reserve(reference.points.size());
  for (const auto & point : reference.points) {
    reference_x.push_back(point.pose.position.x);
    reference_y.push_back(point.pose.position.y);
    reference_acceleration.push_back(
      std::clamp(point.acceleration_mps2, vehicle_params.min_accel(), vehicle_params.max_accel()));
  }
  const auto spline_curvatures =
    computeSmoothedSplineCurvature(reference_x, reference_y, smoothing_weight);
  const auto smoothed_acceleration =
    smoothSecondDifference(reference_acceleration, std::max(0.0F, smoothing_weight));

  for (int t = 0; t < control_count; ++t) {
    const std::size_t index =
      std::min(start_idx + static_cast<std::size_t>(t), reference.points.size() - 1U);
    const auto & point = reference.points[index];
    auto & control = nominal[static_cast<std::size_t>(t)];
    control.accel_cmd = std::clamp(
      smoothed_acceleration[index], vehicle_params.min_accel(), vehicle_params.max_accel());
    float steering = point.front_wheel_angle_rad;
    if (std::abs(steering) <= 1.0E-6F) {
      const float curvature = spline_curvatures[index];
      if (std::isfinite(curvature)) {
        steering = std::atan(vehicle_params.wheel_base * curvature);
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
