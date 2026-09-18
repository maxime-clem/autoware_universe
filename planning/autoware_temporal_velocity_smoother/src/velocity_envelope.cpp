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

#include "autoware/temporal_velocity_smoother/velocity_envelope.hpp"

#include "autoware/temporal_velocity_smoother/warm_start.hpp"

#include <autoware/velocity_smoother/trajectory_utils.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

namespace autoware::temporal_velocity_smoother
{
namespace
{

double table_value(
  const double velocity, const std::vector<double> & thresholds, const std::vector<double> & values,
  const double fallback)
{
  if (thresholds.empty() || thresholds.size() != values.size()) {
    return fallback;
  }
  if (velocity <= thresholds.front()) {
    return values.front();
  }
  for (std::size_t k = 1; k < thresholds.size(); ++k) {
    if (velocity <= thresholds[k]) {
      const double width = thresholds[k] - thresholds[k - 1U];
      if (width <= 1.0e-9) {
        return values[k];
      }
      const double ratio = (velocity - thresholds[k - 1U]) / width;
      return values[k - 1U] + ratio * (values[k] - values[k - 1U]);
    }
  }
  return values.back();
}

}  // namespace

std::vector<double> build_velocity_envelope(
  const ReferencePath & path, const std::vector<double> & query_s, const TrajectoryPoints &,
  const double global_velocity_limit, const double wheel_base,
  const EnvelopeParameters & parameters,
  const std::vector<autoware::trajectory_modifier::SemanticSpeedTracker::SlowSpeedInfo> &
    slow_down_ranges,
  const std::optional<double> & stop_s, const Limits & limits)
{
  std::vector<double> base_cap(path.samples().size(), std::max(0.0, global_velocity_limit));
  std::vector<double> geometry_cap(path.samples().size(), kInfinity);
  for (std::size_t k = 0; k < path.samples().size(); ++k) {
    const auto & point = path.samples()[k];
    const double reference_velocity = std::abs(point.longitudinal_velocity_mps);
    base_cap[k] = std::min(base_cap[k], reference_velocity);
    const double curvature = std::abs(path.curvatures()[k]);
    if (parameters.lateral_acceleration_enabled) {
      const double lateral_limit = table_value(
        reference_velocity, parameters.velocity_thresholds, parameters.lateral_acceleration_limits,
        0.8);
      const double curve_cap = std::max(
        parameters.minimum_curve_velocity,
        std::sqrt(std::max(0.0, lateral_limit) / std::max(curvature, 1.0e-5)));
      geometry_cap[k] = std::min(geometry_cap[k], curve_cap);
    }
    if (parameters.steering_rate_enabled && curvature >= parameters.steering_curvature_threshold) {
      const auto before = k == 0U ? 0U : k - 1U;
      const auto after = std::min(k + 1U, path.samples().size() - 1U);
      const double ds = path.arc_lengths()[after] - path.arc_lengths()[before];
      if (ds > 1.0e-6) {
        const double delta_before = std::atan(wheel_base * path.curvatures()[before]);
        const double delta_after = std::atan(wheel_base * path.curvatures()[after]);
        const double derivative = std::abs(delta_after - delta_before) / ds;
        if (derivative > 1.0e-8) {
          const double rate_deg_s = table_value(
            reference_velocity, parameters.velocity_thresholds,
            parameters.steering_rate_limits_deg_s, 40.0);
          constexpr double degrees_to_radians = 3.14159265358979323846 / 180.0;
          geometry_cap[k] = std::min(geometry_cap[k], rate_deg_s * degrees_to_radians / derivative);
        }
      }
    }
    for (const auto & range : slow_down_ranges) {
      if (path.arc_lengths()[k] >= range.start_s_m && path.arc_lengths()[k] <= range.end_s_m) {
        base_cap[k] = std::min(base_cap[k], parameters.stop_approach_velocity);
      }
    }
  }

  // Extend curve restrictions spatially so the vehicle starts slowing before the curve.
  auto windowed_geometry_cap = geometry_cap;
  for (std::size_t k = 0; k < geometry_cap.size(); ++k) {
    const double low = path.arc_lengths()[k] - parameters.deceleration_distance_after_curve;
    const double high = path.arc_lengths()[k] + parameters.deceleration_distance_before_curve;
    const auto first = std::lower_bound(path.arc_lengths().begin(), path.arc_lengths().end(), low);
    const auto last = std::upper_bound(path.arc_lengths().begin(), path.arc_lengths().end(), high);
    for (auto it = first; it != last; ++it) {
      const auto index = static_cast<std::size_t>(std::distance(path.arc_lengths().begin(), it));
      windowed_geometry_cap[k] = std::min(windowed_geometry_cap[k], geometry_cap[index]);
    }
  }

  auto braking = make_braking_envelope(query_s, stop_s, limits.min_acceleration);
  if (stop_s && parameters.use_jerk_aware_braking_envelope) {
    for (std::size_t k = 0; k < braking.size(); ++k) {
      const double available_distance = std::max(0.0, *stop_s - query_s[k]);
      double low = 0.0;
      double high = braking[k];
      for (int iteration = 0; iteration < 30; ++iteration) {
        const double candidate_velocity = 0.5 * (low + high);
        double stopping_distance = 0.0;
        std::map<double, double> jerk_profile;
        const bool valid =
          autoware::velocity_smoother::trajectory_utils::calcStopDistWithJerkConstraints(
            candidate_velocity, 0.0, limits.max_jerk, limits.min_jerk, limits.min_acceleration, 0.0,
            jerk_profile, stopping_distance);
        if (valid && stopping_distance <= available_distance) {
          low = candidate_velocity;
        } else {
          high = candidate_velocity;
        }
      }
      braking[k] = low;
    }
  }
  std::vector<double> result(query_s.size(), std::max(0.0, global_velocity_limit));
  for (std::size_t k = 0; k < query_s.size(); ++k) {
    const double s = std::clamp(query_s[k], 0.0, path.length());
    const auto it = std::lower_bound(path.arc_lengths().begin(), path.arc_lengths().end(), s);
    const auto index = it == path.arc_lengths().end()
                         ? base_cap.size() - 1U
                         : static_cast<std::size_t>(std::distance(path.arc_lengths().begin(), it));
    result[k] = std::min({base_cap[index], windowed_geometry_cap[index], braking[k]});
    if (stop_s && query_s[k] >= *stop_s - parameters.stop_approach_distance) {
      result[k] = std::min(result[k], parameters.stop_approach_velocity);
    }
  }
  return result;
}

double terminal_safe_velocity(
  const double terminal_s, const double path_end_s, const double minimum_acceleration,
  const double margin)
{
  const double remaining = std::max(0.0, path_end_s - terminal_s - margin);
  return std::sqrt(2.0 * std::abs(minimum_acceleration) * remaining);
}

}  // namespace autoware::temporal_velocity_smoother
