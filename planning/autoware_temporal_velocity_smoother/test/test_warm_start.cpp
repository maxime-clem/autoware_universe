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

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

TEST(WarmStart, IntegratesMonotoneLimitedProfile)
{
  using namespace autoware::temporal_velocity_smoother;
  Limits limits;
  const auto profile = make_warm_start({0.0, 2.0, 0.0}, std::vector<double>(20, 5.0), limits, 0.1);
  ASSERT_EQ(profile.states.size(), 20U);
  ASSERT_EQ(profile.jerk.size(), 20U);
  LongitudinalState previous{0.0, 2.0, 0.0};
  for (std::size_t k = 0; k < profile.states.size(); ++k) {
    EXPECT_NEAR(
      profile.states[k].s,
      previous.s + previous.v * 0.1 + 0.5 * previous.a * 0.01 + profile.jerk[k] * 0.001 / 6.0,
      1.0e-12);
    EXPECT_NEAR(
      profile.states[k].v, previous.v + previous.a * 0.1 + 0.5 * profile.jerk[k] * 0.01, 1.0e-12);
    EXPECT_NEAR(profile.states[k].a, previous.a + profile.jerk[k] * 0.1, 1.0e-12);
    EXPECT_GE(profile.states[k].s, previous.s);
    EXPECT_GE(profile.states[k].v, 0.0);
    EXPECT_LE(profile.states[k].v, 5.0 + 1.0e-9);
    EXPECT_GE(profile.jerk[k], limits.min_jerk - 1.0e-9);
    EXPECT_LE(profile.jerk[k], limits.max_jerk + 1.0e-9);
    previous = profile.states[k];
  }
}

TEST(WarmStart, RecomputesEntireStateWhenVelocityCapIsReachable)
{
  using namespace autoware::temporal_velocity_smoother;
  Limits limits;
  const auto profile = make_warm_start({0.0, 2.0, 0.0}, {1.999}, limits, 0.1);
  ASSERT_EQ(profile.states.size(), 1U);
  ASSERT_EQ(profile.jerk.size(), 1U);
  EXPECT_NEAR(profile.jerk.front(), -0.2, 1.0e-12);
  EXPECT_NEAR(profile.states.front().v, 1.999, 1.0e-12);
  EXPECT_NEAR(profile.states.front().a, -0.02, 1.0e-12);
  EXPECT_NEAR(profile.states.front().s, 0.2 - 0.2 * 0.001 / 6.0, 1.0e-12);
}

TEST(WarmStart, BrakingEnvelopeStopsAtPosition)
{
  using namespace autoware::temporal_velocity_smoother;
  const auto envelope = make_braking_envelope({0.0, 5.0, 10.0, 12.0}, 10.0, -2.0);
  ASSERT_EQ(envelope.size(), 4U);
  EXPECT_NEAR(envelope[0], std::sqrt(40.0), 1.0e-9);
  EXPECT_DOUBLE_EQ(envelope[2], 0.0);
  EXPECT_DOUBLE_EQ(envelope[3], 0.0);
}
