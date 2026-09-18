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

#include "autoware/temporal_velocity_smoother/warm_start.hpp"

#include <algorithm>
#include <cmath>

namespace autoware::temporal_velocity_smoother
{

std::vector<double> make_braking_envelope(
  const std::vector<double> & s_reference, const std::optional<double> & stop_s,
  const double minimum_acceleration)
{
  std::vector<double> envelope(s_reference.size(), kInfinity);
  if (!stop_s) {
    return envelope;
  }
  const double deceleration = std::max(1.0e-6, std::abs(minimum_acceleration));
  for (std::size_t k = 0; k < s_reference.size(); ++k) {
    envelope[k] = std::sqrt(2.0 * deceleration * std::max(0.0, *stop_s - s_reference[k]));
  }
  return envelope;
}

WarmStartProfile make_warm_start(
  const LongitudinalState & initial, const std::vector<double> & velocity_max,
  const Limits & limits, const double dt)
{
  WarmStartProfile profile;
  profile.states.reserve(velocity_max.size());
  profile.jerk.reserve(velocity_max.size());
  auto state = initial;
  for (const double raw_cap : velocity_max) {
    const double cap = std::max(0.0, raw_cap);
    const double desired_acceleration =
      std::clamp((cap - state.v) / dt, limits.min_acceleration, limits.max_acceleration);
    const double jerk =
      std::clamp((desired_acceleration - state.a) / dt, limits.min_jerk, limits.max_jerk);
    LongitudinalState next;
    next.s = state.s + state.v * dt + 0.5 * state.a * dt * dt + jerk * dt * dt * dt / 6.0;
    next.v = std::max(0.0, state.v + state.a * dt + 0.5 * jerk * dt * dt);
    next.a = std::clamp(state.a + jerk * dt, limits.min_acceleration, limits.max_acceleration);
    if (next.v > cap) {
      next.v = cap;
      next.a = std::min(0.0, next.a);
    }
    next.s = std::max(state.s, next.s);
    profile.states.push_back(next);
    profile.jerk.push_back(jerk);
    state = next;
  }
  return profile;
}

}  // namespace autoware::temporal_velocity_smoother
