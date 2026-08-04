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

#include "autoware/mppi_optimizer/detail/trajectory_utils.hpp"
#include "autoware/mppi_optimizer/first_order_dubins_mppi_interface.hpp"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace autoware::mppi_optimizer
{
namespace
{

Trajectory makeStraightTrajectory(const std::size_t point_count)
{
  Trajectory trajectory;
  trajectory.header.frame_id = "map";
  trajectory.header.stamp.sec = 123;
  for (std::size_t i = 0; i < point_count; ++i) {
    autoware_planning_msgs::msg::TrajectoryPoint point;
    point.pose.position.x = 0.2 * static_cast<double>(i + 1U);
    point.pose.position.y = 0.0;
    point.pose.position.z = 1.0 + static_cast<double>(i);
    point.pose.orientation.w = 1.0;
    point.longitudinal_velocity_mps = 2.0F;
    point.time_from_start.sec = static_cast<std::int32_t>(i / 10U);
    point.time_from_start.nanosec = static_cast<std::uint32_t>((i % 10U) * 100000000U);
    trajectory.points.push_back(point);
  }
  return trajectory;
}

Odometry makeOdometry()
{
  Odometry odometry;
  odometry.header.frame_id = "map";
  odometry.pose.pose.orientation.w = 1.0;
  odometry.twist.twist.linear.x = 2.0;
  return odometry;
}

FirstOrderDubinsMppiOptimizationResult optimize(
  FirstOrderDubinsMppiInterface & interface, const Trajectory & trajectory,
  const std::vector<Segment> & road_borders = {})
{
  return interface.optimizeTrajectory(
    trajectory, makeOdometry(), std::nullopt, std::nullopt, TrackedObjects{}, road_borders, {});
}

TEST(FirstOrderDubinsMppiInterface, SkippedInputsDoNotInitializeCuda)
{
  FirstOrderDubinsMppiInterface interface;
  const auto empty_result = optimize(interface, Trajectory{});
  EXPECT_TRUE(empty_result.trajectory.points.empty());
  EXPECT_FALSE(interface.isInitialized());

  auto short_stopping = makeStraightTrajectory(3U);
  short_stopping.points[1].longitudinal_velocity_mps = 0.0F;
  const auto stopping_result = optimize(interface, short_stopping);
  EXPECT_TRUE(stopping_result.trajectory == short_stopping);
  EXPECT_TRUE(stopping_result.debug.reference_trajectory == short_stopping);
  EXPECT_TRUE(stopping_result.debug.optimized_trajectory == short_stopping);
  EXPECT_FALSE(interface.isInitialized());
}

class FirstOrderDubinsMppiInterfaceGpuTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    int device_count = 0;
    const cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "A CUDA device is required for the MPPI integration test";
    }
    interface_ = std::make_unique<FirstOrderDubinsMppiInterface>();
  }

  std::unique_ptr<FirstOrderDubinsMppiInterface> interface_;
};

TEST_F(FirstOrderDubinsMppiInterfaceGpuTest, ProducesFinitePostStepTrajectoryAndPreservesSuffix)
{
  const auto input = makeStraightTrajectory(85U);
  const auto result = optimize(*interface_, input);

  ASSERT_TRUE(interface_->isInitialized());
  ASSERT_EQ(result.trajectory.points.size(), input.points.size());
  EXPECT_EQ(result.trajectory.header, input.header);
  EXPECT_TRUE(result.debug.reference_trajectory == input);
  EXPECT_TRUE(result.debug.optimized_trajectory == result.trajectory);
  EXPECT_TRUE(result.debug.rollouts.empty());
  EXPECT_TRUE(std::isfinite(result.debug.baseline_cost));
  EXPECT_EQ(result.debug.optimal_horizon.size(), static_cast<std::size_t>(detail::kMppiHorizon));

  const auto & first = result.trajectory.points.front();
  EXPECT_NEAR(first.pose.position.x, 0.2, 1.0E-5);
  EXPECT_NEAR(first.pose.position.y, 0.0, 1.0E-5);
  EXPECT_NEAR(first.longitudinal_velocity_mps, 2.0F, 1.0E-5F);
  EXPECT_DOUBLE_EQ(first.pose.position.z, input.points.front().pose.position.z);

  for (std::size_t i = 0; i < static_cast<std::size_t>(detail::kMppiHorizon); ++i) {
    const auto & point = result.trajectory.points[i];
    EXPECT_TRUE(std::isfinite(point.pose.position.x));
    EXPECT_TRUE(std::isfinite(point.pose.position.y));
    EXPECT_TRUE(std::isfinite(point.longitudinal_velocity_mps));
    EXPECT_TRUE(std::isfinite(point.acceleration_mps2));
    EXPECT_TRUE(std::isfinite(point.front_wheel_angle_rad));
    EXPECT_LE(std::abs(point.acceleration_mps2), 7.0F + 1.0E-5F);
    EXPECT_LE(std::abs(point.front_wheel_angle_rad), 0.45F + 1.0E-5F);
  }
  for (std::size_t i = static_cast<std::size_t>(detail::kMppiHorizon); i < input.points.size();
       ++i) {
    EXPECT_TRUE(result.trajectory.points[i] == input.points[i]);
  }
}

TEST_F(FirstOrderDubinsMppiInterfaceGpuTest, ReturnsInputWhenFirstPostStepHitsRoadBorder)
{
  FirstOrderDubinsMppiRuntimeOptions options;
  options.skip_if_invalid = true;
  interface_->setRuntimeOptions(options);

  const auto input = makeStraightTrajectory(30U);
  const Segment crossing_border{0.2F, -2.0F, 0.2F, 2.0F};
  const auto result = optimize(*interface_, input, {crossing_border});

  EXPECT_TRUE(interface_->isInitialized());
  EXPECT_TRUE(result.trajectory == input);
  EXPECT_TRUE(result.debug.reference_trajectory == input);
  EXPECT_TRUE(result.debug.optimized_trajectory == input);
  EXPECT_TRUE(std::isfinite(result.debug.baseline_cost));
}

}  // namespace
}  // namespace autoware::mppi_optimizer
