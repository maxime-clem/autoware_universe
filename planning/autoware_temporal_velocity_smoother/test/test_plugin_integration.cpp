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

#include "autoware/trajectory_modifier/trajectory_modifier_plugin_base.hpp"

#include <pluginlib/class_loader.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <string>

TEST(PluginIntegration, IsDiscoverableAndConstructible)
{
  using Base = autoware::trajectory_modifier::plugin::TrajectoryModifierPluginBase;
  constexpr auto class_name =
    "autoware::temporal_velocity_smoother::plugin::TemporalVelocitySmoother";
  pluginlib::ClassLoader<Base> loader(
    "autoware_trajectory_modifier",
    "autoware::trajectory_modifier::plugin::TrajectoryModifierPluginBase");
  const auto classes = loader.getDeclaredClasses();
  EXPECT_NE(std::find(classes.begin(), classes.end(), class_name), classes.end());
  EXPECT_NE(loader.createUniqueInstance(class_name), nullptr);
}
