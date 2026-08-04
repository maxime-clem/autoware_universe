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

#include <autoware_perception_msgs/autoware_perception_msgs/msg/detail/tracked_objects__struct.hpp>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
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
  const Odometry & odometry = makeOdometry(),
  const TrackedObjects & tracked_objects = TrackedObjects{},
  const std::vector<Segment> & road_borders = {})
{
  return interface.optimizeTrajectory(
    trajectory, odometry, std::nullopt, std::nullopt, tracked_objects, road_borders, {});
}

TrackedObjects makeStationaryBoxObstacle(
  const double x, const double y, const double length, const double width)
{
  TrackedObjects objects;
  objects.objects.emplace_back();
  auto & object = objects.objects.back();
  object.kinematics.pose_with_covariance.pose.position.x = x;
  object.kinematics.pose_with_covariance.pose.position.y = y;
  object.kinematics.pose_with_covariance.pose.orientation.w = 1.0;
  object.shape.dimensions.x = length;
  object.shape.dimensions.y = width;
  return objects;
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
  EXPECT_TRUE(result.debug.validation.isValid());
  EXPECT_FALSE(result.debug.was_rejected);
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

TEST_F(FirstOrderDubinsMppiInterfaceGpuTest, RejectsAtInclusiveLateralBoundaryThreshold)
{
  FirstOrderDubinsMppiCostParams cost_params;
  cost_params.boundary_threshold = 0.5F;
  interface_->setCostParams(cost_params);
  FirstOrderDubinsMppiRuntimeOptions options;
  options.skip_if_invalid = true;
  interface_->setRuntimeOptions(options);

  const auto input = makeStraightTrajectory(30U);
  auto odometry = makeOdometry();
  odometry.pose.pose.position.y = 0.5;
  const auto result = optimize(*interface_, input, odometry);

  EXPECT_TRUE(interface_->isInitialized());
  EXPECT_TRUE(result.trajectory == input);
  EXPECT_TRUE(result.debug.reference_trajectory == input);
  EXPECT_TRUE(result.debug.optimized_trajectory == input);
  EXPECT_TRUE(result.debug.was_rejected);
  EXPECT_TRUE(hasInvalidityReason(
    result.debug.validation.reasons, FirstOrderDubinsMppiInvalidityReason::lateral_boundary));
  ASSERT_TRUE(result.debug.validation.first_invalid_index.has_value());
  EXPECT_EQ(result.debug.validation.first_invalid_index.value(), 0U);
  EXPECT_TRUE(std::isfinite(result.debug.baseline_cost));
}

TEST_F(FirstOrderDubinsMppiInterfaceGpuTest, RejectsRoadBorderInsideConfiguredMargin)
{
  FirstOrderDubinsMppiCostParams cost_params;
  cost_params.boundary_threshold = 100.0F;
  cost_params.road_border_collision_margin = 0.2F;
  interface_->setCostParams(cost_params);
  FirstOrderDubinsMppiRuntimeOptions options;
  options.skip_if_invalid = true;
  interface_->setRuntimeOptions(options);

  const auto input = makeStraightTrajectory(30U);
  const Segment border_outside_physical_footprint{-1.0F, 0.31F, 2.0F, 0.31F};
  const auto result = optimize(
    *interface_, input, makeOdometry(), TrackedObjects{}, {border_outside_physical_footprint});

  EXPECT_TRUE(result.trajectory == input);
  EXPECT_TRUE(result.debug.was_rejected);
  EXPECT_TRUE(hasInvalidityReason(
    result.debug.validation.reasons, FirstOrderDubinsMppiInvalidityReason::road_border));
  ASSERT_TRUE(result.debug.validation.first_invalid_index.has_value());
  EXPECT_EQ(result.debug.validation.first_invalid_index.value(), 0U);
}

TEST_F(FirstOrderDubinsMppiInterfaceGpuTest, RejectsObjectInsideConfiguredMargin)
{
  FirstOrderDubinsMppiCostParams cost_params;
  cost_params.boundary_threshold = 100.0F;
  cost_params.obstacle_collision_margin = 0.2F;
  interface_->setCostParams(cost_params);
  FirstOrderDubinsMppiRuntimeOptions options;
  options.skip_if_invalid = true;
  interface_->setRuntimeOptions(options);

  const auto input = makeStraightTrajectory(30U);
  const auto objects = makeStationaryBoxObstacle(0.4, 0.41, 0.2, 0.2);
  const auto result = optimize(*interface_, input, makeOdometry(), objects);

  EXPECT_TRUE(result.trajectory == input);
  EXPECT_TRUE(result.debug.was_rejected);
  EXPECT_TRUE(hasInvalidityReason(
    result.debug.validation.reasons, FirstOrderDubinsMppiInvalidityReason::obstacle));
  ASSERT_TRUE(result.debug.validation.first_invalid_index.has_value());
  EXPECT_EQ(result.debug.validation.first_invalid_index.value(), 0U);
}

TEST_F(FirstOrderDubinsMppiInterfaceGpuTest, DoesNotRejectInvalidOutputWhenOptionIsDisabled)
{
  FirstOrderDubinsMppiCostParams cost_params;
  cost_params.boundary_threshold = 0.5F;
  interface_->setCostParams(cost_params);
  FirstOrderDubinsMppiRuntimeOptions options;
  options.skip_if_invalid = false;
  interface_->setRuntimeOptions(options);

  const auto input = makeStraightTrajectory(30U);
  auto odometry = makeOdometry();
  odometry.pose.pose.position.y = 0.5;
  const auto result = optimize(*interface_, input, odometry);

  EXPECT_FALSE(result.debug.validation.isValid());
  EXPECT_TRUE(hasInvalidityReason(
    result.debug.validation.reasons, FirstOrderDubinsMppiInvalidityReason::lateral_boundary));
  EXPECT_FALSE(result.debug.was_rejected);
  EXPECT_FALSE(result.trajectory == input);
  EXPECT_TRUE(result.debug.optimized_trajectory == result.trajectory);
}

TEST_F(FirstOrderDubinsMppiInterfaceGpuTest, HandlesInitialOffsetsAcrossThresholdRange)
{
  FirstOrderDubinsMppiRuntimeOptions options;
  options.skip_if_invalid = true;
  interface_->setRuntimeOptions(options);

  struct TestCase
  {
    float threshold;
    float initial_y_offset;
    bool expect_rejected;
  };

  const std::vector<TestCase> test_cases = {
    {0.50F, 0.30F, false}, {0.50F, -0.30F, false}, {0.50F, 0.50F, true}, {0.50F, -0.50F, true},
    {0.50F, 0.60F, true},  {0.50F, -0.60F, true},  {0.10F, 0.11F, true},
  };

  for (std::size_t idx = 0; idx < test_cases.size(); ++idx) {
    const auto & tc = test_cases[idx];
    FirstOrderDubinsMppiCostParams cost_params;
    cost_params.boundary_threshold = tc.threshold;
    interface_->setCostParams(cost_params);

    const auto input = makeStraightTrajectory(30U);
    auto odometry = makeOdometry();
    odometry.pose.pose.position.y = tc.initial_y_offset;

    const auto result = optimize(*interface_, input, odometry, TrackedObjects{}, {});

    EXPECT_EQ(result.debug.was_rejected, tc.expect_rejected)
      << "Failed at test case index " << idx << " (threshold=" << tc.threshold
      << ", init_y=" << tc.initial_y_offset << ")";

    EXPECT_EQ(result.debug.validation.isValid(), !tc.expect_rejected)
      << "Validation mismatch at test case index " << idx;

    if (tc.expect_rejected) {
      EXPECT_TRUE(result.trajectory == input)
        << "Rejected trajectory was not properly reverted to input at index " << idx;
      EXPECT_TRUE(hasInvalidityReason(
        result.debug.validation.reasons, FirstOrderDubinsMppiInvalidityReason::lateral_boundary))
        << "Missing lateral_boundary reason bit at index " << idx;
    }
  }
}

}  // namespace
}  // namespace autoware::mppi_optimizer
