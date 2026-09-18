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

#include "autoware/temporal_velocity_smoother/temporal_velocity_smoother_plugin.hpp"

#include "autoware/temporal_velocity_smoother/reference_path.hpp"
#include "autoware/temporal_velocity_smoother/velocity_envelope.hpp"
#include "autoware/temporal_velocity_smoother/warm_start.hpp"
#include "autoware/trajectory_modifier/utils.hpp"

#include <autoware_utils_debug/time_keeper.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/duration.hpp>
#include <rclcpp/time.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace autoware::temporal_velocity_smoother::plugin
{
namespace
{

using autoware::trajectory_modifier::plugin::ProcessingResult;
using autoware::trajectory_modifier::plugin::TrajectoryPoints;

double seconds(const builtin_interfaces::msg::Duration & duration)
{
  return static_cast<double>(duration.sec) + 1.0e-9 * static_cast<double>(duration.nanosec);
}

Weights make_weights(const ::temporal_velocity_smoother::Params & params)
{
  Weights weights;
  weights.velocity_tracking = params.weights.velocity_tracking;
  weights.acceleration_tracking = params.weights.accel_tracking;
  weights.jerk = params.weights.jerk;
  weights.progress = params.weights.progress;
  weights.over_velocity = params.weights.over_v;
  weights.over_acceleration = params.weights.over_a;
  weights.over_jerk = params.weights.over_j;
  weights.over_position = params.weights.over_s;
  return weights;
}

EnvelopeParameters make_envelope_parameters(const ::temporal_velocity_smoother::Params & params)
{
  EnvelopeParameters output;
  output.lateral_acceleration_enabled = params.lateral_acc.enable;
  output.velocity_thresholds = params.lateral_acc.velocity_thresholds;
  output.lateral_acceleration_limits = params.lateral_acc.limits;
  output.minimum_curve_velocity = params.lateral_acc.min_curve_velocity;
  output.deceleration_distance_before_curve = params.lateral_acc.decel_distance_before_curve;
  output.deceleration_distance_after_curve = params.lateral_acc.decel_distance_after_curve;
  output.steering_rate_enabled = params.steer_rate.enable;
  output.steering_rate_limits_deg_s = params.steer_rate.limits_deg_s;
  output.steering_curvature_threshold = params.steer_rate.curvature_threshold;
  output.stop_approach_velocity = params.stop.stopping_velocity;
  output.stop_approach_distance = params.stop.stopping_distance;
  output.use_jerk_aware_braking_envelope = params.use_jerk_aware_braking_envelope;
  return output;
}

std::optional<double> find_stop_position(
  const TrajectoryPoints & points, const ReferencePath & path)
{
  for (std::size_t k = 0; k < points.size() && k < path.input_arc_lengths().size(); ++k) {
    if (std::abs(points[k].longitudinal_velocity_mps) < 1.0e-3) {
      return path.input_arc_lengths()[k];
    }
  }
  return std::nullopt;
}

bool validate_input(const TrajectoryPoints & points, const double dt)
{
  if (points.size() < 2U || !(dt > 0.0)) {
    return false;
  }
  for (const auto & point : points) {
    if (!autoware::trajectory_modifier::utils::validate_point(point)) {
      return false;
    }
    if (point.longitudinal_velocity_mps < -1.0e-3) {
      return false;
    }
  }
  if (std::abs(seconds(points.front().time_from_start) - dt) > 0.25 * dt) {
    return false;
  }
  for (std::size_t k = 0; k + 1U < points.size(); ++k) {
    const double point_dt =
      autoware::trajectory_modifier::utils::compute_dt(points[k], points[k + 1U]);
    if (!std::isfinite(point_dt) || std::abs(point_dt - dt) > 0.25 * dt) {
      return false;
    }
  }
  return true;
}

}  // namespace

void TemporalVelocitySmoother::apply_shared_params(
  const autoware::trajectory_modifier::TrajectoryModifierParams & params)
{
  trajectory_time_step_ = params.trajectory_time_step;
  default_max_velocity_ = params.max_vel;
  limits_.max_velocity = params.max_vel;
  limits_.max_acceleration = params.limit.max_acc;
  limits_.min_acceleration = params.limit.min_acc;
  limits_.max_jerk = params.limit.max_jerk;
  limits_.min_jerk = params.limit.min_jerk;
}

void TemporalVelocitySmoother::on_initialize(
  const autoware::trajectory_modifier::TrajectoryModifierParams & params)
{
  auto * const node = get_node_ptr();
  param_listener_ = std::make_unique<::temporal_velocity_smoother::ParamListener>(
    node, "temporal_velocity_smoother");
  params_ = param_listener_->get_params();
  apply_shared_params(params);
  enabled_ = params_.enabled;
  velocity_limit_subscriber_ =
    std::make_shared<autoware_utils_rclcpp::InterProcessPollingSubscriber<
      autoware_internal_planning_msgs::msg::VelocityLimit>>(
      node, "~/input/external_velocity_limit_mps", rclcpp::QoS{1});
  debug_trajectory_publisher_ = node->create_publisher<autoware_planning_msgs::msg::Trajectory>(
    "~/debug/temporal_velocity_smoother/trajectory", 1);
}

void TemporalVelocitySmoother::update_params(
  const autoware::trajectory_modifier::TrajectoryModifierParams & params)
{
  apply_shared_params(params);
  auto updated = params_;
  if (param_listener_->try_update_params(updated)) {
    params_ = std::move(updated);
    enabled_ = params_.enabled;
    candidates_.clear();
  }
}

void TemporalVelocitySmoother::reset_candidates(const std::size_t count)
{
  candidates_.clear();
  candidates_.resize(count);
  for (auto & candidate : candidates_) {
    candidate.qp = std::make_unique<LongitudinalQp>(
      static_cast<std::size_t>(params_.horizon_points), params_.solver.enable_warm_start,
      static_cast<int>(params_.solver.max_iteration), params_.solver.eps_abs,
      params_.solver.eps_rel, false);
  }
}

LongitudinalState TemporalVelocitySmoother::select_initial_state(
  const CandidateState & state, const autoware::trajectory_modifier::TrajectoryModifierData & data,
  const TrajectoryPoints & trajectory) const
{
  LongitudinalState initial{
    0.0, std::max(0.0, data.current_odometry->twist.twist.linear.x),
    data.current_acceleration->accel.accel.linear.x};
  initial.a = std::clamp(initial.a, limits_.min_acceleration, limits_.max_acceleration);

  if (state.has_previous_plan) {
    const double age =
      (rclcpp::Time(data.candidate_header.stamp) - rclcpp::Time(state.previous_stamp)).seconds();
    if (age >= 0.0 && age <= params_.initial_state.max_plan_age && !state.previous_plan.empty()) {
      const double scaled = age / trajectory_time_step_;
      const auto high =
        std::min(static_cast<std::size_t>(std::ceil(scaled)), state.previous_plan.size());
      LongitudinalState planned;
      if (high == 0U) {
        planned = state.previous_initial;
      } else {
        const auto low_state = high == 1U ? state.previous_initial : state.previous_plan[high - 2U];
        const auto high_state = state.previous_plan[high - 1U];
        const double ratio = std::clamp(scaled - static_cast<double>(high - 1U), 0.0, 1.0);
        planned.v = low_state.v + ratio * (high_state.v - low_state.v);
        planned.a = low_state.a + ratio * (high_state.a - low_state.a);
      }
      if (std::abs(planned.v - initial.v) <= params_.initial_state.replan_vel_deviation) {
        initial.v = std::max(0.0, planned.v);
        initial.a = std::clamp(planned.a, limits_.min_acceleration, limits_.max_acceleration);
      }
    }
  }

  if (
    initial.v < params_.initial_state.engage_velocity * params_.initial_state.engage_exit_ratio &&
    !trajectory.empty() &&
    trajectory.front().longitudinal_velocity_mps > params_.initial_state.engage_velocity) {
    initial.v = params_.initial_state.engage_velocity;
    initial.a = params_.initial_state.engage_acceleration;
  }
  return initial;
}

ProcessingResult TemporalVelocitySmoother::process(
  TrajectoryPoints & trajectory_points,
  autoware::trajectory_modifier::TrajectoryModifierData & data)
{
  autoware_utils_debug::ScopedTimeTrack time_track(__func__, *get_time_keeper());
  auto updated = params_;
  if (param_listener_->try_update_params(updated)) {
    params_ = std::move(updated);
    enabled_ = params_.enabled;
    candidates_.clear();
  }
  if (
    !enabled_ || (params_.only_primary_candidate && data.candidate_index != 0U) ||
    !data.current_odometry || !data.current_acceleration) {
    return ProcessingResult::Unchanged;
  }
  if (!validate_input(trajectory_points, trajectory_time_step_)) {
    RCLCPP_WARN_THROTTLE(
      get_node_ptr()->get_logger(), *get_clock(), 5000,
      "TemporalVelocitySmoother rejected an invalid, reverse, or non-constant-time trajectory");
    return ProcessingResult::Unchanged;
  }
  if (params_.sqp_iterations == 1 && params_.terminal_stop_margin < 1.0) {
    RCLCPP_ERROR_THROTTLE(
      get_node_ptr()->get_logger(), *get_clock(), 5000,
      "sqp_iterations=1 requires terminal_stop_margin >= 1.0 m");
    return ProcessingResult::Unchanged;
  }
  const std::size_t candidate_count = std::max<std::size_t>(1U, data.candidate_count);
  if (candidates_.size() != candidate_count) {
    reset_candidates(candidate_count);
  }
  if (data.candidate_index >= candidates_.size()) {
    return ProcessingResult::Unchanged;
  }
  auto & candidate = candidates_[data.candidate_index];

  const ReferencePath path(
    data.current_odometry->pose.pose, trajectory_points, params_.path.resample_ds,
    params_.path.curvature_calculation_distance);
  if (!path.valid()) {
    return ProcessingResult::Unchanged;
  }

  const auto input_points = trajectory_points;
  const std::size_t horizon = static_cast<std::size_t>(params_.horizon_points);
  std::vector<double> reference_s(horizon, path.length());
  std::vector<double> reference_v(horizon, 0.0);
  std::vector<double> reference_a(horizon, 0.0);
  std::vector<bool> tracking_enabled(horizon, false);
  for (std::size_t k = 0; k < horizon; ++k) {
    if (k < path.input_arc_lengths().size()) {
      reference_s[k] = path.input_arc_lengths()[k];
    }
    const auto point = path.sample(reference_s[k]);
    reference_v[k] = std::max(0.0, static_cast<double>(point.longitudinal_velocity_mps));
    reference_a[k] = point.acceleration_mps2;
    tracking_enabled[k] = k < input_points.size();
  }

  double max_acceleration_disagreement = 0.0;
  for (std::size_t k = 0; k + 1U < input_points.size(); ++k) {
    const double finite_difference =
      (input_points[k + 1U].longitudinal_velocity_mps - input_points[k].longitudinal_velocity_mps) /
      trajectory_time_step_;
    max_acceleration_disagreement = std::max(
      max_acceleration_disagreement,
      std::abs(finite_difference - input_points[k].acceleration_mps2));
  }
  if (max_acceleration_disagreement > 1.0) {
    RCLCPP_DEBUG(
      get_node_ptr()->get_logger(), "Input velocity/acceleration disagreement: %.3f m/s^2",
      max_acceleration_disagreement);
  }

  double global_limit = default_max_velocity_;
  if (const auto external_limit = velocity_limit_subscriber_->take_data()) {
    global_limit = std::min(global_limit, static_cast<double>(external_limit->max_velocity));
  }
  const auto stop_s = find_stop_position(input_points, path);
  const double effective_end = stop_s ? std::min(path.length(), *stop_s) : path.length();
  const auto envelope_parameters = make_envelope_parameters(params_);
  const auto initial = select_initial_state(candidate, data, input_points);

  std::vector<double> last_envelope = build_velocity_envelope(
    path, reference_s, input_points, global_limit, context_->vehicle_info.wheel_base_m,
    envelope_parameters, data.semantic_speed_tracker.get_slow_down_ranges(), stop_s, limits_);
  const auto fallback_profile =
    make_warm_start(initial, last_envelope, limits_, trajectory_time_step_);
  std::vector<double> linearization_s;
  linearization_s.reserve(horizon);
  for (const auto & state : fallback_profile.states) {
    linearization_s.push_back(state.s);
  }

  SolveResult solved;
  const int sqp_iterations = std::max(1, static_cast<int>(params_.sqp_iterations));
  for (int iteration = 0; iteration < sqp_iterations; ++iteration) {
    last_envelope = build_velocity_envelope(
      path, linearization_s, input_points, global_limit, context_->vehicle_info.wheel_base_m,
      envelope_parameters, data.semantic_speed_tracker.get_slow_down_ranges(), stop_s, limits_);
    QpInput qp_input;
    qp_input.dt = trajectory_time_step_;
    qp_input.initial = initial;
    qp_input.limits = limits_;
    qp_input.weights = make_weights(params_);
    qp_input.reference_s = reference_s;
    qp_input.reference_v = reference_v;
    qp_input.reference_a = reference_a;
    qp_input.velocity_max = last_envelope;
    qp_input.tracking_enabled = tracking_enabled;
    if (
      stop_s || params_.path_end_policy == "clamp_s" ||
      params_.path_end_policy == "shrink_horizon") {
      qp_input.position_limit = effective_end;
    }
    qp_input.terminal_velocity_limit = terminal_safe_velocity(
      linearization_s.back(), effective_end, limits_.min_acceleration,
      params_.terminal_stop_margin);
    try {
      solved = candidate.qp->solve(qp_input);
    } catch (const std::exception & error) {
      solved = SolveResult{};
      solved.status = error.what();
    }
    if (!solved.success) {
      break;
    }
    double maximum_change = 0.0;
    for (std::size_t k = 0; k < horizon; ++k) {
      maximum_change = std::max(maximum_change, std::abs(solved.states[k].s - linearization_s[k]));
      linearization_s[k] = solved.states[k].s;
    }
    if (maximum_change < params_.sqp_position_tolerance) {
      break;
    }
  }

  std::vector<LongitudinalState> output_states;
  if (solved.success) {
    output_states = solved.states;
  } else {
    output_states = fallback_profile.states;
    RCLCPP_WARN_THROTTLE(
      get_node_ptr()->get_logger(), *get_clock(), 5000,
      "TemporalVelocitySmoother QP failed (%s); publishing integrated fallback",
      solved.status.c_str());
  }

  TrajectoryPoints output;
  output.reserve(horizon);
  std::vector<double> output_s;
  output_s.reserve(horizon);
  const bool extrapolate = params_.path_end_policy == "extrapolate";
  for (std::size_t k = 0; k < horizon; ++k) {
    auto state = output_states[k];
    if (!extrapolate) {
      state.s = std::clamp(state.s, 0.0, effective_end);
    }
    if (stop_s && state.s >= *stop_s - 1.0e-3) {
      state.s = *stop_s;
      state.v = 0.0;
      state.a = params_.stop.stop_decel;
    }
    auto point = path.sample(state.s, extrapolate);
    point.longitudinal_velocity_mps = static_cast<float>(std::max(0.0, state.v));
    point.acceleration_mps2 = static_cast<float>(state.a);
    point.front_wheel_angle_rad =
      static_cast<float>(std::atan(context_->vehicle_info.wheel_base_m * path.curvature(state.s)));
    point.time_from_start =
      rclcpp::Duration::from_seconds(static_cast<double>(k + 1U) * trajectory_time_step_);
    output_states[k] = state;
    output.push_back(point);
    if (output.size() == 1U) {
      output_s.push_back(0.0);
    } else {
      const auto & previous = output[output.size() - 2U].pose.position;
      const auto & current = output.back().pose.position;
      output_s.push_back(
        output_s.back() + std::hypot(current.x - previous.x, current.y - previous.y));
    }
  }
  if (params_.path_end_policy == "shrink_horizon") {
    while (output.size() > 2U && output_s[output.size() - 2U] >= effective_end - 1.0e-6) {
      output.pop_back();
      output_s.pop_back();
    }
  }

  data.semantic_speed_tracker.remap_to_trajectory(output_s);
  candidate.previous_initial = initial;
  candidate.previous_plan = output_states;
  candidate.previous_stamp = data.candidate_header.stamp;
  candidate.has_previous_plan = true;

  if (params_.publish_debug_trajectories) {
    autoware_planning_msgs::msg::Trajectory debug;
    debug.header = data.candidate_header;
    debug.points = output;
    pending_debug_trajectory_ = std::move(debug);
  }
  if (!params_.shadow_mode) {
    trajectory_points = std::move(output);
    return ProcessingResult::Modified;
  }
  return ProcessingResult::Unchanged;
}

void TemporalVelocitySmoother::publish_debug_data(const std::string &) const
{
  if (pending_debug_trajectory_) {
    debug_trajectory_publisher_->publish(*pending_debug_trajectory_);
    pending_debug_trajectory_.reset();
  }
}

}  // namespace autoware::temporal_velocity_smoother::plugin

PLUGINLIB_EXPORT_CLASS(
  autoware::temporal_velocity_smoother::plugin::TemporalVelocitySmoother,
  autoware::trajectory_modifier::plugin::TrajectoryModifierPluginBase)
