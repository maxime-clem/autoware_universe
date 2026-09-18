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

#ifndef AUTOWARE__TEMPORAL_VELOCITY_SMOOTHER__TYPES_HPP_
#define AUTOWARE__TEMPORAL_VELOCITY_SMOOTHER__TYPES_HPP_

#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace autoware::temporal_velocity_smoother
{

struct LongitudinalState
{
  double s{0.0};
  double v{0.0};
  double a{0.0};
};

struct Limits
{
  double max_velocity{11.1};
  double max_acceleration{1.0};
  double min_acceleration{-2.5};
  double max_jerk{1.5};
  double min_jerk{-1.5};
};

struct Weights
{
  double velocity_tracking{1.0};
  double acceleration_tracking{10.0};
  double jerk{30.0};
  double progress{0.0};
  double over_velocity{3000.0};
  double over_acceleration{30.0};
  double over_jerk{10.0};
  double over_position{100000.0};
};

struct ProblemLayout
{
  explicit ProblemLayout(const std::size_t horizon)
  : n(horizon),
    s(0U),
    v(n),
    a(2U * n),
    jerk(3U * n),
    velocity_slack(4U * n),
    acceleration_slack(5U * n),
    jerk_slack(6U * n),
    position_slack(7U * n),
    variables(8U * n),
    dynamics_row(0U),
    velocity_row(3U * n),
    acceleration_row(4U * n),
    jerk_row(5U * n),
    position_row(6U * n),
    monotonic_row(7U * n),
    terminal_row(8U * n - 1U),
    constraints(8U * n)
  {
  }

  std::size_t n;
  std::size_t s;
  std::size_t v;
  std::size_t a;
  std::size_t jerk;
  std::size_t velocity_slack;
  std::size_t acceleration_slack;
  std::size_t jerk_slack;
  std::size_t position_slack;
  std::size_t variables;
  std::size_t dynamics_row;
  std::size_t velocity_row;
  std::size_t acceleration_row;
  std::size_t jerk_row;
  std::size_t position_row;
  std::size_t monotonic_row;
  std::size_t terminal_row;
  std::size_t constraints;
};

struct QpInput
{
  double dt{0.1};
  LongitudinalState initial;
  Limits limits;
  Weights weights;
  std::vector<double> reference_s;
  std::vector<double> reference_v;
  std::vector<double> reference_a;
  std::vector<double> velocity_max;
  std::vector<bool> tracking_enabled;
  std::optional<double> position_limit;
  std::optional<double> terminal_velocity_limit;
  bool allow_reverse{false};
};

struct SolveResult
{
  bool success{false};
  std::string status;
  int iterations{0};
  std::vector<LongitudinalState> states;
  std::vector<double> jerk;
  std::vector<double> velocity_slack;
  std::vector<double> acceleration_slack;
  std::vector<double> jerk_slack;
  std::vector<double> position_slack;
};

struct WarmStartProfile
{
  std::vector<LongitudinalState> states;
  std::vector<double> jerk;
};

inline constexpr double kInfinity = std::numeric_limits<double>::infinity();

}  // namespace autoware::temporal_velocity_smoother

#endif  // AUTOWARE__TEMPORAL_VELOCITY_SMOOTHER__TYPES_HPP_
