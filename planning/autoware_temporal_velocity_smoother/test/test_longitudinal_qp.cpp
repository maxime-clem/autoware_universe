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

#include "autoware/temporal_velocity_smoother/longitudinal_qp.hpp"

#include <gtest/gtest.h>

#include <vector>

TEST(LongitudinalQp, HasFixedDimensionsAndSatisfiesDynamics)
{
  using namespace autoware::temporal_velocity_smoother;
  constexpr std::size_t n = 12U;
  constexpr double dt = 0.1;
  LongitudinalQp qp(n, true, 20000, 1.0e-8, 1.0e-6);
  EXPECT_EQ(qp.layout().variables, 8U * n);
  EXPECT_EQ(qp.layout().constraints, 8U * n);

  QpInput input;
  input.dt = dt;
  input.initial = {0.0, 2.0, 0.0};
  input.reference_s.resize(n);
  input.reference_v.assign(n, 2.0);
  input.reference_a.assign(n, 0.0);
  input.velocity_max.assign(n, 5.0);
  input.tracking_enabled.assign(n, true);
  for (std::size_t k = 0; k < n; ++k) {
    input.reference_s[k] = static_cast<double>(k + 1U) * dt * 2.0;
  }
  input.terminal_velocity_limit = 5.0;
  const auto result = qp.solve(input);
  ASSERT_TRUE(result.success) << result.status;
  ASSERT_EQ(result.states.size(), n);
  LongitudinalState previous = input.initial;
  for (std::size_t k = 0; k < n; ++k) {
    EXPECT_NEAR(
      result.states[k].s,
      previous.s + previous.v * dt + 0.5 * previous.a * dt * dt +
        result.jerk[k] * dt * dt * dt / 6.0,
      1.0e-5);
    EXPECT_NEAR(
      result.states[k].v, previous.v + previous.a * dt + 0.5 * result.jerk[k] * dt * dt, 1.0e-5);
    EXPECT_NEAR(result.states[k].a, previous.a + result.jerk[k] * dt, 1.0e-5);
    previous = result.states[k];
  }
}
