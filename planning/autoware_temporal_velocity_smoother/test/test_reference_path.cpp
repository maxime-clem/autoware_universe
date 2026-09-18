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

#include <gtest/gtest.h>

#include <vector>

TEST(ReferencePath, AnchorsAtEgoAndSamplesPath)
{
  using namespace autoware::temporal_velocity_smoother;
  geometry_msgs::msg::Pose ego;
  ego.orientation.w = 1.0;
  TrajectoryPoints points(2);
  points[0].pose.position.x = 1.0;
  points[0].pose.orientation.w = 1.0;
  points[0].longitudinal_velocity_mps = 2.0;
  points[1] = points[0];
  points[1].pose.position.x = 3.0;
  ReferencePath path(ego, points, 0.1, 1.0);
  ASSERT_TRUE(path.valid());
  EXPECT_NEAR(path.length(), 3.0, 1.0e-9);
  EXPECT_NEAR(path.sample(0.0).pose.position.x, 0.0, 1.0e-9);
  EXPECT_NEAR(path.sample(1.5).pose.position.x, 1.5, 1.0e-6);
  EXPECT_NEAR(path.curvature(1.5), 0.0, 1.0e-9);
  EXPECT_NEAR(path.sample(4.0, true).pose.position.x, 4.0, 1.0e-6);
}
