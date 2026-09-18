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

#ifndef AUTOWARE__TEMPORAL_VELOCITY_SMOOTHER__REFERENCE_PATH_HPP_
#define AUTOWARE__TEMPORAL_VELOCITY_SMOOTHER__REFERENCE_PATH_HPP_

#include <autoware_planning_msgs/msg/trajectory_point.hpp>
#include <geometry_msgs/msg/pose.hpp>

#include <cstddef>
#include <vector>

namespace autoware::temporal_velocity_smoother
{

using TrajectoryPoint = autoware_planning_msgs::msg::TrajectoryPoint;
using TrajectoryPoints = std::vector<TrajectoryPoint>;

class ReferencePath
{
public:
  ReferencePath() = default;
  ReferencePath(
    const geometry_msgs::msg::Pose & ego_pose, const TrajectoryPoints & points,
    double resample_distance, double curvature_calculation_distance);

  [[nodiscard]] bool valid() const { return samples_.size() >= 2U; }
  [[nodiscard]] double length() const { return arc_lengths_.empty() ? 0.0 : arc_lengths_.back(); }
  [[nodiscard]] const std::vector<double> & arc_lengths() const { return arc_lengths_; }
  [[nodiscard]] const std::vector<double> & input_arc_lengths() const { return input_arc_lengths_; }
  [[nodiscard]] const TrajectoryPoints & samples() const { return samples_; }
  [[nodiscard]] const std::vector<double> & curvatures() const { return curvatures_; }

  [[nodiscard]] TrajectoryPoint sample(double s, bool extrapolate = false) const;
  [[nodiscard]] double curvature(double s) const;

private:
  TrajectoryPoints samples_;
  std::vector<double> arc_lengths_;
  std::vector<double> input_arc_lengths_;
  std::vector<double> curvatures_;
};

}  // namespace autoware::temporal_velocity_smoother

#endif  // AUTOWARE__TEMPORAL_VELOCITY_SMOOTHER__REFERENCE_PATH_HPP_
