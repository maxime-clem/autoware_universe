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

#include <gtest/gtest.h>

#include <vector>

TEST(VelocityEnvelope, CombinesGlobalReferenceAndStopCaps)
{
  using namespace autoware::temporal_velocity_smoother;
  geometry_msgs::msg::Pose ego;
  ego.orientation.w = 1.0;
  TrajectoryPoints points(2);
  points[0].pose.position.x = 1.0;
  points[0].pose.orientation.w = 1.0;
  points[0].longitudinal_velocity_mps = 8.0;
  points[1] = points[0];
  points[1].pose.position.x = 10.0;
  ReferencePath path(ego, points, 0.5, 2.0);
  EnvelopeParameters parameters;
  parameters.lateral_acceleration_enabled = false;
  parameters.steering_rate_enabled = false;
  Limits limits;
  const auto envelope =
    build_velocity_envelope(path, {0.0, 5.0, 10.0}, points, 6.0, 2.7, parameters, {}, 10.0, limits);
  ASSERT_EQ(envelope.size(), 3U);
  EXPECT_LE(envelope[0], 6.0);
  EXPECT_LE(envelope[1], 6.0);
  EXPECT_DOUBLE_EQ(envelope[2], 0.0);
}
