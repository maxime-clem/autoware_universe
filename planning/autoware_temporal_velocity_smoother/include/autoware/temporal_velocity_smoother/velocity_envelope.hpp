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

#ifndef AUTOWARE__TEMPORAL_VELOCITY_SMOOTHER__VELOCITY_ENVELOPE_HPP_
#define AUTOWARE__TEMPORAL_VELOCITY_SMOOTHER__VELOCITY_ENVELOPE_HPP_

#include "autoware/temporal_velocity_smoother/reference_path.hpp"
#include "autoware/temporal_velocity_smoother/types.hpp"

#include <autoware/trajectory_modifier/semantic_speed_tracker.hpp>

#include <optional>
#include <vector>

namespace autoware::temporal_velocity_smoother
{

struct EnvelopeParameters
{
  bool lateral_acceleration_enabled{true};
  std::vector<double> velocity_thresholds{0.1, 0.3, 20.0, 30.0};
  std::vector<double> lateral_acceleration_limits{0.8, 0.8, 0.8, 0.8};
  double minimum_curve_velocity{2.74};
  double deceleration_distance_before_curve{3.5};
  double deceleration_distance_after_curve{2.0};
  bool steering_rate_enabled{true};
  std::vector<double> steering_rate_limits_deg_s{57.0, 57.0, 57.0, 40.0};
  double steering_curvature_threshold{0.02};
  double stop_approach_velocity{2.778};
  double stop_approach_distance{0.0};
  bool use_jerk_aware_braking_envelope{false};
};

std::vector<double> build_velocity_envelope(
  const ReferencePath & path, const std::vector<double> & query_s,
  const TrajectoryPoints & incoming, double global_velocity_limit, double wheel_base,
  const EnvelopeParameters & parameters,
  const std::vector<autoware::trajectory_modifier::SemanticSpeedTracker::SlowSpeedInfo> &
    slow_down_ranges,
  const std::optional<double> & stop_s, const Limits & limits);

double terminal_safe_velocity(
  double terminal_s, double path_end_s, double minimum_acceleration, double margin);

}  // namespace autoware::temporal_velocity_smoother

#endif  // AUTOWARE__TEMPORAL_VELOCITY_SMOOTHER__VELOCITY_ENVELOPE_HPP_
