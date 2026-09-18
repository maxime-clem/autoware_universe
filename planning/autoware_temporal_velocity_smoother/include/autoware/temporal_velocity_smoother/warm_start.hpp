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

#ifndef AUTOWARE__TEMPORAL_VELOCITY_SMOOTHER__WARM_START_HPP_
#define AUTOWARE__TEMPORAL_VELOCITY_SMOOTHER__WARM_START_HPP_

#include "autoware/temporal_velocity_smoother/types.hpp"

#include <cstddef>
#include <optional>
#include <vector>

namespace autoware::temporal_velocity_smoother
{

std::vector<double> make_braking_envelope(
  const std::vector<double> & s_reference, const std::optional<double> & stop_s,
  double minimum_acceleration);

WarmStartProfile make_warm_start(
  const LongitudinalState & initial, const std::vector<double> & velocity_max,
  const Limits & limits, double dt);

}  // namespace autoware::temporal_velocity_smoother

#endif  // AUTOWARE__TEMPORAL_VELOCITY_SMOOTHER__WARM_START_HPP_
