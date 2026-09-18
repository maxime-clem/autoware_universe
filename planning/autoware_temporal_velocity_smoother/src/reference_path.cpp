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

#include "autoware/temporal_velocity_smoother/reference_path.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace autoware::temporal_velocity_smoother
{
namespace
{

double distance(const TrajectoryPoint & lhs, const TrajectoryPoint & rhs)
{
  return std::hypot(
    lhs.pose.position.x - rhs.pose.position.x, lhs.pose.position.y - rhs.pose.position.y);
}

TrajectoryPoint interpolate(
  const TrajectoryPoint & lhs, const TrajectoryPoint & rhs, const double ratio)
{
  TrajectoryPoint output = lhs;
  const auto lerp = [ratio](const double a, const double b) { return a + ratio * (b - a); };
  output.pose.position.x = lerp(lhs.pose.position.x, rhs.pose.position.x);
  output.pose.position.y = lerp(lhs.pose.position.y, rhs.pose.position.y);
  output.pose.position.z = lerp(lhs.pose.position.z, rhs.pose.position.z);
  output.pose.orientation.x = lerp(lhs.pose.orientation.x, rhs.pose.orientation.x);
  output.pose.orientation.y = lerp(lhs.pose.orientation.y, rhs.pose.orientation.y);
  output.pose.orientation.z = lerp(lhs.pose.orientation.z, rhs.pose.orientation.z);
  output.pose.orientation.w = lerp(lhs.pose.orientation.w, rhs.pose.orientation.w);
  const double norm = std::sqrt(
    output.pose.orientation.x * output.pose.orientation.x +
    output.pose.orientation.y * output.pose.orientation.y +
    output.pose.orientation.z * output.pose.orientation.z +
    output.pose.orientation.w * output.pose.orientation.w);
  if (norm > 1.0e-9) {
    output.pose.orientation.x /= norm;
    output.pose.orientation.y /= norm;
    output.pose.orientation.z /= norm;
    output.pose.orientation.w /= norm;
  }
  output.longitudinal_velocity_mps =
    static_cast<float>(lerp(lhs.longitudinal_velocity_mps, rhs.longitudinal_velocity_mps));
  output.lateral_velocity_mps =
    static_cast<float>(lerp(lhs.lateral_velocity_mps, rhs.lateral_velocity_mps));
  output.acceleration_mps2 = static_cast<float>(lerp(lhs.acceleration_mps2, rhs.acceleration_mps2));
  output.heading_rate_rps = static_cast<float>(lerp(lhs.heading_rate_rps, rhs.heading_rate_rps));
  output.front_wheel_angle_rad =
    static_cast<float>(lerp(lhs.front_wheel_angle_rad, rhs.front_wheel_angle_rad));
  output.rear_wheel_angle_rad =
    static_cast<float>(lerp(lhs.rear_wheel_angle_rad, rhs.rear_wheel_angle_rad));
  return output;
}

TrajectoryPoint sample_polyline(
  const TrajectoryPoints & points, const std::vector<double> & arcs, const double s)
{
  if (s <= 0.0) {
    return points.front();
  }
  if (s >= arcs.back()) {
    return points.back();
  }
  const auto upper = std::upper_bound(arcs.begin(), arcs.end(), s);
  const auto high = static_cast<std::size_t>(std::distance(arcs.begin(), upper));
  const auto low = high - 1U;
  const double span = arcs[high] - arcs[low];
  return interpolate(points[low], points[high], (s - arcs[low]) / span);
}

double three_point_curvature(
  const TrajectoryPoint & first, const TrajectoryPoint & middle, const TrajectoryPoint & last)
{
  const double ax = middle.pose.position.x - first.pose.position.x;
  const double ay = middle.pose.position.y - first.pose.position.y;
  const double bx = last.pose.position.x - middle.pose.position.x;
  const double by = last.pose.position.y - middle.pose.position.y;
  const double cx = last.pose.position.x - first.pose.position.x;
  const double cy = last.pose.position.y - first.pose.position.y;
  const double denom = std::hypot(ax, ay) * std::hypot(bx, by) * std::hypot(cx, cy);
  if (denom < 1.0e-8) {
    return 0.0;
  }
  return 2.0 * (ax * cy - ay * cx) / denom;
}

}  // namespace

ReferencePath::ReferencePath(
  const geometry_msgs::msg::Pose & ego_pose, const TrajectoryPoints & points,
  const double resample_distance, const double curvature_calculation_distance)
{
  if (points.empty() || !(resample_distance > 0.0)) {
    return;
  }

  TrajectoryPoints raw;
  std::vector<double> raw_arcs;
  TrajectoryPoint ego_point = points.front();
  ego_point.pose = ego_pose;
  raw.push_back(ego_point);
  raw_arcs.push_back(0.0);
  input_arc_lengths_.reserve(points.size());
  double input_s = 0.0;
  TrajectoryPoint previous = ego_point;
  for (const auto & point : points) {
    input_s += distance(previous, point);
    input_arc_lengths_.push_back(input_s);
    if (distance(raw.back(), point) > 1.0e-4) {
      raw.push_back(point);
      raw_arcs.push_back(input_s);
    }
    previous = point;
  }
  if (raw.size() < 2U || raw_arcs.back() < 1.0e-4) {
    return;
  }

  for (double s = 0.0; s < raw_arcs.back(); s += resample_distance) {
    samples_.push_back(sample_polyline(raw, raw_arcs, s));
    arc_lengths_.push_back(s);
  }
  if (arc_lengths_.empty() || raw_arcs.back() - arc_lengths_.back() > 1.0e-6) {
    samples_.push_back(raw.back());
    arc_lengths_.push_back(raw_arcs.back());
  }

  curvatures_.resize(samples_.size(), 0.0);
  const double half_chord = std::max(resample_distance, 0.5 * curvature_calculation_distance);
  for (std::size_t k = 0; k < samples_.size(); ++k) {
    const double center = arc_lengths_[k];
    const auto before = std::lower_bound(
      arc_lengths_.begin(), arc_lengths_.end(), std::max(0.0, center - half_chord));
    const auto after = std::lower_bound(
      arc_lengths_.begin(), arc_lengths_.end(), std::min(length(), center + half_chord));
    const auto i0 = static_cast<std::size_t>(std::distance(arc_lengths_.begin(), before));
    const auto i2 = std::min(
      samples_.size() - 1U, static_cast<std::size_t>(std::distance(arc_lengths_.begin(), after)));
    if (i0 < k && k < i2) {
      curvatures_[k] = three_point_curvature(samples_[i0], samples_[k], samples_[i2]);
    } else if (i2 > i0 + 1U) {
      const auto mid = i0 + (i2 - i0) / 2U;
      curvatures_[k] = three_point_curvature(samples_[i0], samples_[mid], samples_[i2]);
    }
  }
}

TrajectoryPoint ReferencePath::sample(const double requested_s, const bool extrapolate) const
{
  if (!valid()) {
    throw std::runtime_error("cannot sample an invalid reference path");
  }
  if (requested_s <= length() || !extrapolate) {
    return sample_polyline(samples_, arc_lengths_, std::clamp(requested_s, 0.0, length()));
  }
  const auto & previous = samples_[samples_.size() - 2U];
  auto output = samples_.back();
  const double segment_length = distance(previous, output);
  if (segment_length > 1.0e-6) {
    const double extension = requested_s - length();
    output.pose.position.x +=
      extension * (output.pose.position.x - previous.pose.position.x) / segment_length;
    output.pose.position.y +=
      extension * (output.pose.position.y - previous.pose.position.y) / segment_length;
    output.pose.position.z +=
      extension * (output.pose.position.z - previous.pose.position.z) / segment_length;
  }
  return output;
}

double ReferencePath::curvature(const double requested_s) const
{
  if (!valid()) {
    return 0.0;
  }
  const double s = std::clamp(requested_s, 0.0, length());
  const auto upper = std::lower_bound(arc_lengths_.begin(), arc_lengths_.end(), s);
  if (upper == arc_lengths_.begin()) {
    return curvatures_.front();
  }
  if (upper == arc_lengths_.end()) {
    return curvatures_.back();
  }
  const auto high = static_cast<std::size_t>(std::distance(arc_lengths_.begin(), upper));
  const auto low = high - 1U;
  const double ratio = (s - arc_lengths_[low]) / (arc_lengths_[high] - arc_lengths_[low]);
  return curvatures_[low] + ratio * (curvatures_[high] - curvatures_[low]);
}

}  // namespace autoware::temporal_velocity_smoother
