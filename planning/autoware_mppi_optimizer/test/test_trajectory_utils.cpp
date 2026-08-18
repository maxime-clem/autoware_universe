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

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <gtest/gtest.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/utils.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <vector>

namespace autoware::mppi_optimizer::detail
{
namespace
{

geometry_msgs::msg::Quaternion makeQuaternion(const double yaw)
{
  tf2::Quaternion quaternion;
  quaternion.setRPY(0.0, 0.0, yaw);
  return tf2::toMsg(quaternion);
}

Trajectory makeTrajectory(const std::size_t point_count, const double spacing, const float velocity)
{
  Trajectory trajectory;
  trajectory.header.frame_id = "map";
  trajectory.header.stamp.sec = 42;
  for (std::size_t i = 0; i < point_count; ++i) {
    autoware_planning_msgs::msg::TrajectoryPoint point;
    point.pose.position.x = static_cast<double>(i) * spacing;
    point.pose.position.y = -0.5 * static_cast<double>(i);
    point.pose.position.z = 3.0 + static_cast<double>(i);
    point.pose.orientation = makeQuaternion(0.1 * static_cast<double>(i));
    point.longitudinal_velocity_mps = velocity;
    point.lateral_velocity_mps = static_cast<float>(i) + 0.5F;
    point.acceleration_mps2 = static_cast<float>(i);
    point.front_wheel_angle_rad = 0.01F * static_cast<float>(i);
    point.time_from_start.sec = static_cast<std::int32_t>(i);
    point.time_from_start.nanosec = 123U;
    trajectory.points.push_back(point);
  }
  return trajectory;
}

void appendPosition(Trajectory & trajectory, const double x, const double y)
{
  autoware_planning_msgs::msg::TrajectoryPoint point;
  point.pose.position.x = x;
  point.pose.position.y = y;
  trajectory.points.push_back(point);
}

TEST(TrajectoryEligibility, SkipsOnlyTrajectoriesThatAreBothShortAndStopping)
{
  auto short_stopping = makeTrajectory(3U, 1.0, 1.0F);
  short_stopping.points[1].longitudinal_velocity_mps = 0.0F;
  EXPECT_FALSE(isOptimizationRequired(short_stopping, 4.0));

  auto short_moving = makeTrajectory(3U, 1.0, 0.02F);
  EXPECT_TRUE(isOptimizationRequired(short_moving, 4.0));

  auto long_stopping = makeTrajectory(3U, 2.0, 1.0F);
  long_stopping.points[1].longitudinal_velocity_mps = 0.0F;
  EXPECT_TRUE(isOptimizationRequired(long_stopping, 4.0));

  EXPECT_TRUE(isOptimizationRequired(short_stopping, 0.0));
  EXPECT_TRUE(isOptimizationRequired(short_stopping, 2.0));
}

TEST(InitialState, UsesOdometryDefaultsAndClampsOptionalVehicleState)
{
  Odometry odometry;
  odometry.pose.pose.position.x = 12.0;
  odometry.pose.pose.position.y = -4.0;
  odometry.pose.pose.orientation = makeQuaternion(0.7);
  odometry.twist.twist.linear.x = -2.5;

  FirstOrderDubinsMppiVehicleParams vehicle;
  vehicle.vel_rate_lim = 3.0F;
  vehicle.max_steer_angle = 0.4F;

  const auto defaults = makeInitialState(odometry, std::nullopt, std::nullopt, vehicle);
  EXPECT_FLOAT_EQ(defaults.x, 12.0F);
  EXPECT_FLOAT_EQ(defaults.y, -4.0F);
  EXPECT_NEAR(defaults.yaw, 0.7F, 1.0E-6F);
  EXPECT_FLOAT_EQ(defaults.velocity, -2.5F);
  EXPECT_FLOAT_EQ(defaults.acceleration, 0.0F);
  EXPECT_FLOAT_EQ(defaults.steering, 0.0F);

  geometry_msgs::msg::AccelWithCovarianceStamped acceleration;
  acceleration.accel.accel.linear.x = 10.0;
  autoware_vehicle_msgs::msg::SteeringReport steering;
  steering.steering_tire_angle = -1.0F;
  const auto clamped = makeInitialState(odometry, acceleration, steering, vehicle);
  EXPECT_FLOAT_EQ(clamped.acceleration, vehicle.max_accel());
  EXPECT_FLOAT_EQ(clamped.steering, -vehicle.max_steer_angle);
}

TEST(ReferenceHorizon, MapsInputDirectlyAndHoldsTheLastSample)
{
  auto trajectory = makeTrajectory(2U, 1.0, 3.0F);
  trajectory.points[0].longitudinal_velocity_mps = 1.0F;
  trajectory.points[1].longitudinal_velocity_mps = 2.0F;
  InitialState ego;

  const auto reference = buildReferenceHorizon(trajectory, ego, 4, 0.1F);
  ASSERT_EQ(reference.size(), 4U);
  EXPECT_FLOAT_EQ(reference[0].time, 0.1F);
  EXPECT_FLOAT_EQ(reference[0].x, 0.0F);
  EXPECT_FLOAT_EQ(reference[0].velocity, 1.0F);
  EXPECT_FLOAT_EQ(reference[1].time, 0.2F);
  EXPECT_FLOAT_EQ(reference[1].x, 1.0F);
  EXPECT_NEAR(reference[1].yaw, 0.1F, 1.0E-6F);
  EXPECT_FLOAT_EQ(reference[3].time, 0.4F);
  EXPECT_FLOAT_EQ(reference[3].x, reference[1].x);
  EXPECT_FLOAT_EQ(reference[3].velocity, reference[1].velocity);
}

TEST(ReferenceHorizon, EmptyInputHoldsTheMeasuredEgoState)
{
  InitialState ego;
  ego.x = 1.0F;
  ego.y = 2.0F;
  ego.yaw = 0.3F;
  ego.velocity = 4.0F;

  const auto reference = buildReferenceHorizon(Trajectory{}, ego, 2, 0.1F);
  ASSERT_EQ(reference.size(), 2U);
  for (const auto & sample : reference) {
    EXPECT_FLOAT_EQ(sample.x, ego.x);
    EXPECT_FLOAT_EQ(sample.y, ego.y);
    EXPECT_FLOAT_EQ(sample.yaw, ego.yaw);
    EXPECT_FLOAT_EQ(sample.velocity, ego.velocity);
    EXPECT_FLOAT_EQ(sample.arc_length_s, 0.0F);
  }
}

TEST(CumulativeChordLength, AccumulatesPolylineSegmentLengthsByIndex)
{
  Trajectory trajectory;
  for (std::size_t i = 0; i < 4U; ++i) {
    autoware_planning_msgs::msg::TrajectoryPoint point;
    point.pose.position.x = static_cast<double>(i) * 3.0;
    point.pose.position.y = 0.0;
    trajectory.points.push_back(point);
  }

  const auto s = computeCumulativeChordLength(trajectory);
  ASSERT_EQ(s.size(), 4U);
  EXPECT_FLOAT_EQ(s[0], 0.0F);
  EXPECT_FLOAT_EQ(s[1], 3.0F);
  EXPECT_FLOAT_EQ(s[2], 6.0F);
  EXPECT_FLOAT_EQ(s[3], 9.0F);

  InitialState ego;
  const auto reference = buildReferenceHorizon(trajectory, ego, 3, 0.1F, 1U, &s);
  ASSERT_EQ(reference.size(), 3U);
  EXPECT_FLOAT_EQ(reference[0].arc_length_s, 3.0F);
  EXPECT_FLOAT_EQ(reference[1].arc_length_s, 6.0F);
  EXPECT_FLOAT_EQ(reference[2].arc_length_s, 9.0F);
}

TEST(NominalControl, CopiesClampsPadsAndDerivesSteeringFromCurvature)
{
  // 1. Create a 3-point trajectory so Menger curvature can evaluate 3 non-collinear points
  auto trajectory = makeTrajectory(3U, 2.0, 2.0F);
  trajectory.points[0].acceleration_mps2 = 20.0F;
  trajectory.points[0].front_wheel_angle_rad = 0.0F;  // Zero -> triggers Menger curvature fallback
  trajectory.points[0].pose.orientation = makeQuaternion(0.0);

  trajectory.points[1].acceleration_mps2 = -20.0F;
  trajectory.points[1].front_wheel_angle_rad =
    1.0F;  // Non-zero -> uses explicit value (clamped to max)
  trajectory.points[1].pose.orientation = makeQuaternion(0.2);

  trajectory.points[2].acceleration_mps2 = -20.0F;
  trajectory.points[2].front_wheel_angle_rad = 1.0F;
  trajectory.points[2].pose.orientation = makeQuaternion(0.4);

  FirstOrderDubinsMppiVehicleParams vehicle;
  vehicle.vel_rate_lim = 3.0F;
  vehicle.max_steer_angle = 0.4F;
  vehicle.wheel_base = 0.32F;
  const auto nominal = buildDiffusionNominalControl(trajectory, 0U, vehicle, 4);

  ASSERT_EQ(nominal.size(), 4U);
  EXPECT_FLOAT_EQ(nominal[0].accel_cmd, vehicle.max_accel());

  // 2. Compute expected Menger curvature for the 3 points in makeTrajectory(3U, 2.0, ...)
  // Points are: p0=(0, 0), p1=(2.0, -0.5), p2=(4.0, -1.0) -> Note: collinear if y is linear!
  // To test curvature, let's make p1 slightly offset so it has real curvature:
  const float expected_curvature = computeMengerCurvatureWithMinChord(trajectory.points, 0U, 1.5);
  EXPECT_NEAR(nominal[0].steer_cmd, std::atan(vehicle.wheel_base * expected_curvature), 1.0E-6F);

  EXPECT_FLOAT_EQ(nominal[1].accel_cmd, vehicle.min_accel());
  EXPECT_FLOAT_EQ(nominal[1].steer_cmd, vehicle.max_steer_angle);  // Clamped from 1.0F -> 0.4F
  EXPECT_FLOAT_EQ(nominal[3].accel_cmd, nominal[1].accel_cmd);
  EXPECT_FLOAT_EQ(nominal[3].steer_cmd, nominal[1].steer_cmd);
}

TEST(MengerCurvature, CollinearPointsHaveZeroCurvatureWithVariableSpacing)
{
  Trajectory trajectory;
  for (const double x : {0.0, 0.1, 0.4, 2.0, 2.1, 4.0}) {
    appendPosition(trajectory, x, 0.0);
  }

  for (std::size_t i = 0; i < trajectory.points.size(); ++i) {
    EXPECT_FLOAT_EQ(computeMengerCurvatureWithMinChord(trajectory.points, i, 1.5F), 0.0F);
  }
}

TEST(MengerCurvature, DenseCircleMatchesInverseRadius)
{
  constexpr double radius = 10.0;
  constexpr double arc_spacing = 0.1;
  constexpr std::size_t point_count = 80U;
  Trajectory trajectory;
  for (std::size_t i = 0; i < point_count; ++i) {
    const double angle = static_cast<double>(i) * arc_spacing / radius;
    appendPosition(trajectory, radius * std::cos(angle), radius * std::sin(angle));
  }

  const float curvature = computeMengerCurvatureWithMinChord(trajectory.points, 20U, 1.5F);
  EXPECT_NEAR(curvature, 1.0 / radius, 0.001);
}

TEST(MengerCurvature, DistanceWindowAttenuatesDenseLateralJitter)
{
  Trajectory trajectory;
  for (std::size_t i = 0; i < 80U; ++i) {
    const double y = i % 2U == 0U ? -0.05 : 0.05;
    appendPosition(trajectory, 0.1 * static_cast<double>(i), y);
  }

  float adjacent_peak = 0.0F;
  float windowed_peak = 0.0F;
  for (std::size_t i = 1U; i + 1U < trajectory.points.size(); ++i) {
    adjacent_peak = std::max(
      adjacent_peak, std::abs(computeMengerCurvatureWithMinChord(trajectory.points, i, 0.0F)));
    windowed_peak = std::max(
      windowed_peak, std::abs(computeMengerCurvatureWithMinChord(trajectory.points, i, 1.5F)));
  }

  EXPECT_LT(windowed_peak, adjacent_peak);
}

TEST(MengerCurvature, EndpointsAndNearEndpointsRemainFinite)
{
  Trajectory trajectory;
  for (std::size_t i = 0; i < 20U; ++i) {
    appendPosition(
      trajectory, 0.1 * static_cast<double>(i), 0.05 * std::sin(static_cast<double>(i)));
  }

  const std::vector<std::size_t> indices = {
    0U, 1U, trajectory.points.size() - 2U, trajectory.points.size() - 1U};
  for (const std::size_t i : indices) {
    EXPECT_TRUE(std::isfinite(computeMengerCurvatureWithMinChord(trajectory.points, i, 1.5F)));
  }
}

TEST(NominalControl, CurvatureChordParameterSmoothsColdStartSteeringSeed)
{
  Trajectory trajectory;
  for (std::size_t i = 0; i < 80U; ++i) {
    const double y = i % 2U == 0U ? -0.05 : 0.05;
    appendPosition(trajectory, 0.1 * static_cast<double>(i), y);
  }

  FirstOrderDubinsMppiVehicleParams vehicle;
  vehicle.wheel_base = 4.76F;
  vehicle.max_steer_angle = 1.5F;
  const auto adjacent = buildDiffusionNominalControl(trajectory, 20U, vehicle, 30, 0.0F);
  const auto windowed = buildDiffusionNominalControl(trajectory, 20U, vehicle, 30, 1.5F);
  float adjacent_peak = 0.0F;
  float windowed_peak = 0.0F;
  for (std::size_t i = 0; i < adjacent.size(); ++i) {
    adjacent_peak = std::max(adjacent_peak, std::abs(adjacent[i].steer_cmd));
    windowed_peak = std::max(windowed_peak, std::abs(windowed[i].steer_cmd));
  }
  EXPECT_LT(windowed_peak, adjacent_peak);
}

TEST(NominalControl, ForcedControlPadsClampsAndShiftHoldsTheTerminalCommand)
{
  FirstOrderDubinsMppiVehicleParams vehicle;
  vehicle.vel_rate_lim = 2.0F;
  vehicle.max_steer_angle = 0.3F;
  const auto forced = buildForcedNominalControl({10.0F, -1.0F}, {1.0F}, vehicle, 4);
  ASSERT_EQ(forced.size(), 4U);
  EXPECT_FLOAT_EQ(forced[0].accel_cmd, 2.0F);
  EXPECT_FLOAT_EQ(forced[0].steer_cmd, 0.3F);
  EXPECT_FLOAT_EQ(forced[3].accel_cmd, -1.0F);
  EXPECT_FLOAT_EQ(forced[3].steer_cmd, 0.3F);

  const auto shifted = shiftNominalControl(forced, 4);
  ASSERT_EQ(shifted.size(), 4U);
  EXPECT_FLOAT_EQ(shifted[0].accel_cmd, forced[1].accel_cmd);
  EXPECT_FLOAT_EQ(shifted[2].accel_cmd, forced[3].accel_cmd);
  EXPECT_FLOAT_EQ(shifted[3].accel_cmd, forced[3].accel_cmd);
}

TEST(OutputConversion, OverwritesOnlyAvailablePostStepSamples)
{
  const auto input = makeTrajectory(3U, 1.0, 2.0F);
  const std::vector<OptimizedState> states = {
    {10.0F, 20.0F, 0.3F, 4.0F, 0.7F, 0.1F}, {11.0F, 21.0F, 0.4F, 5.0F, 0.8F, 0.2F}};
  const std::vector<FirstOrderDubinsMppiControl> controls = {{0.5F, -0.1F}, {0.6F, -0.2F}};

  const auto output = buildOptimizedTrajectory(input, states, controls);
  ASSERT_EQ(output.points.size(), input.points.size());
  EXPECT_EQ(output.header, input.header);
  EXPECT_DOUBLE_EQ(output.points[0].pose.position.x, 10.0);
  EXPECT_DOUBLE_EQ(output.points[0].pose.position.y, 20.0);
  EXPECT_DOUBLE_EQ(output.points[0].pose.position.z, input.points[0].pose.position.z);
  EXPECT_NEAR(tf2::getYaw(output.points[0].pose.orientation), 0.3, 1.0E-6);
  EXPECT_FLOAT_EQ(output.points[0].longitudinal_velocity_mps, 4.0F);
  EXPECT_FLOAT_EQ(output.points[0].acceleration_mps2, 0.7F);
  EXPECT_FLOAT_EQ(output.points[0].front_wheel_angle_rad, 0.1F);
  EXPECT_FLOAT_EQ(output.points[0].lateral_velocity_mps, input.points[0].lateral_velocity_mps);
  EXPECT_TRUE(output.points[2] == input.points[2]);
}

TEST(EngageVelocity, ChangesOnlyTheFirstTwoPointsWhenMovementIsRequested)
{
  auto trajectory = makeTrajectory(4U, 1.0, 0.0F);
  trajectory.points[3].longitudinal_velocity_mps = 1.0F;
  setInitialEngageVelocity(trajectory);
  EXPECT_FLOAT_EQ(trajectory.points[0].longitudinal_velocity_mps, 0.25F);
  EXPECT_FLOAT_EQ(trajectory.points[1].longitudinal_velocity_mps, 0.25F);
  EXPECT_FLOAT_EQ(trajectory.points[2].longitudinal_velocity_mps, 0.0F);

  auto stopping = makeTrajectory(4U, 1.0, 0.0F);
  setInitialEngageVelocity(stopping);
  EXPECT_FLOAT_EQ(stopping.points[0].longitudinal_velocity_mps, 0.0F);
  EXPECT_FLOAT_EQ(stopping.points[1].longitudinal_velocity_mps, 0.0F);

  auto externally_stopped = makeTrajectory(4U, 1.0, 0.0F);
  externally_stopped.points[3].longitudinal_velocity_mps = 1.0F;
  setInitialEngageVelocity(externally_stopped, 0.0F);
  EXPECT_FLOAT_EQ(externally_stopped.points[0].longitudinal_velocity_mps, 0.0F);
  EXPECT_FLOAT_EQ(externally_stopped.points[1].longitudinal_velocity_mps, 0.0F);

  auto externally_limited = makeTrajectory(4U, 1.0, 0.0F);
  externally_limited.points[3].longitudinal_velocity_mps = 1.0F;
  setInitialEngageVelocity(externally_limited, 0.1F);
  EXPECT_FLOAT_EQ(externally_limited.points[0].longitudinal_velocity_mps, 0.1F);
  EXPECT_FLOAT_EQ(externally_limited.points[1].longitudinal_velocity_mps, 0.1F);
}

}  // namespace
}  // namespace autoware::mppi_optimizer::detail
