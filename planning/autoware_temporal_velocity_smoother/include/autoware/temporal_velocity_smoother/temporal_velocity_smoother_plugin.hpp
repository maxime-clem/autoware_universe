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

#ifndef AUTOWARE__TEMPORAL_VELOCITY_SMOOTHER__TEMPORAL_VELOCITY_SMOOTHER_PLUGIN_HPP_
#define AUTOWARE__TEMPORAL_VELOCITY_SMOOTHER__TEMPORAL_VELOCITY_SMOOTHER_PLUGIN_HPP_

#include "autoware/temporal_velocity_smoother/longitudinal_qp.hpp"
#include "autoware/temporal_velocity_smoother/types.hpp"
#include "autoware/trajectory_modifier/trajectory_modifier_plugin_base.hpp"

#include <autoware_temporal_velocity_smoother/temporal_velocity_smoother_parameters.hpp>
#include <autoware_utils_rclcpp/polling_subscriber.hpp>

#include <autoware_internal_planning_msgs/msg/velocity_limit.hpp>
#include <autoware_planning_msgs/msg/trajectory.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace autoware::temporal_velocity_smoother::plugin
{

class TemporalVelocitySmoother final
: public autoware::trajectory_modifier::plugin::TrajectoryModifierPluginBase
{
public:
  autoware::trajectory_modifier::plugin::ProcessingResult process(
    autoware::trajectory_modifier::plugin::TrajectoryPoints & trajectory_points,
    autoware::trajectory_modifier::TrajectoryModifierData & data) override;

  void update_params(
    const autoware::trajectory_modifier::TrajectoryModifierParams & params) override;

  void publish_debug_data(const std::string & ns) const override;

protected:
  void on_initialize(
    const autoware::trajectory_modifier::TrajectoryModifierParams & params) override;

private:
  struct CandidateState
  {
    std::unique_ptr<LongitudinalQp> qp;
    LongitudinalState previous_initial;
    std::vector<LongitudinalState> previous_plan;
    builtin_interfaces::msg::Time previous_stamp;
    bool has_previous_plan{false};
  };

  void apply_shared_params(const autoware::trajectory_modifier::TrajectoryModifierParams & params);
  void reset_candidates(std::size_t count);
  [[nodiscard]] LongitudinalState select_initial_state(
    const CandidateState & state,
    const autoware::trajectory_modifier::TrajectoryModifierData & data,
    const autoware::trajectory_modifier::plugin::TrajectoryPoints & trajectory) const;

  std::unique_ptr<::temporal_velocity_smoother::ParamListener> param_listener_;
  ::temporal_velocity_smoother::Params params_;
  Limits limits_;
  double default_max_velocity_{11.1};
  std::vector<CandidateState> candidates_;
  std::shared_ptr<autoware_utils_rclcpp::InterProcessPollingSubscriber<
    autoware_internal_planning_msgs::msg::VelocityLimit>>
    velocity_limit_subscriber_;
  rclcpp::Publisher<autoware_planning_msgs::msg::Trajectory>::SharedPtr debug_trajectory_publisher_;
  mutable std::optional<autoware_planning_msgs::msg::Trajectory> pending_debug_trajectory_;
};

}  // namespace autoware::temporal_velocity_smoother::plugin

#endif  // AUTOWARE__TEMPORAL_VELOCITY_SMOOTHER__TEMPORAL_VELOCITY_SMOOTHER_PLUGIN_HPP_
