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
    const double acceleration_jerk_min = (limits.min_acceleration - state.a) / dt;
    const double acceleration_jerk_max = (limits.max_acceleration - state.a) / dt;
    const double jerk_min = std::max(limits.min_jerk, acceleration_jerk_min);
    const double jerk_max = std::min(limits.max_jerk, acceleration_jerk_max);
    const double jerk_to_cap = 2.0 * (cap - state.v - state.a * dt) / (dt * dt);
    const double jerk = std::clamp(jerk_to_cap, jerk_min, jerk_max);
    LongitudinalState next;
    next.s = state.s + state.v * dt + 0.5 * state.a * dt * dt + jerk * dt * dt * dt / 6.0;
    next.v = state.v + state.a * dt + 0.5 * jerk * dt * dt;
    next.a = state.a + jerk * dt;

    // Prevent reverse motion while maintaining physical consistency at a stop
    if (next.v < 0.0) {
      next.v = 0.0;
      next.a = 0.0;  // The vehicle has stopped; acceleration vanishes

      // Calculate exactly how long it took to hit 0 velocity within this dt interval
      // Using quadratic formula: 0.5 * jerk * t^2 + a * t + v = 0
      // For a simple linear approximation when jerk is small: t ≈ -state.v / state.a
      // Or simply hold the position at the zero-crossing estimate:
      next.s = state.s + (-0.5 * state.v * state.v) / std::min(-1e-6, state.a);
    }

    profile.states.push_back(next);
    profile.jerk.push_back(jerk);
    state = next;
  }
  return profile;
}

}  // namespace autoware::temporal_velocity_smoother
