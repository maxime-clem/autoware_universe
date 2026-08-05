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

#include "autoware/mppi_optimizer/detail/trajectory_validator.hpp"

#include <mppi/cost_functions/dubins/first_order_dubins_bicycle_cost.cuh>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace autoware::mppi_optimizer
{
namespace
{

constexpr int kTestHorizon = detail::kMppiHorizon;
using TestCost = FirstOrderDubinsBicycleCost<kTestHorizon>;
using TestCostParams = FirstOrderDubinsBicycleCostParams<kTestHorizon>;
using CostOutputIndex = FirstOrderDubinsBicycleParams::OutputIndex;
using CostControlIndex = FirstOrderDubinsBicycleParams::ControlIndex;

TEST(FirstOrderDubinsBicycleSteeringCost, ZeroCoefficientsPreserveLegacyComfortCost)
{
  TestCost cost;
  TestCostParams params;
  params.steer_rate_l2_coeff = 0.0F;
  params.steer_accel_coeff = 0.0F;
  params.cmd_slew_coeff = 0.0F;
  cost.setParams(params);

  TestCost::control_array control = TestCost::control_array::Zero();
  TestCost::output_array output = TestCost::output_array::Zero();
  control(static_cast<int>(CostControlIndex::STEER_CMD)) = 0.2F;
  output(static_cast<int>(CostOutputIndex::BASELINK_VEL_B_X)) = 2.0F;
  output(static_cast<int>(CostOutputIndex::STEER_ANGLE)) = 0.1F;
  output(static_cast<int>(CostOutputIndex::ACCELERATION)) = 0.3F;

  cost.setLastAppliedSteerCommand(-0.4F);
  const float first_cost = cost.computeComfortCost(control, output, 0);
  output(static_cast<int>(CostOutputIndex::STEER_RATE)) = 3.0F;
  output(static_cast<int>(CostOutputIndex::PREVIOUS_STEER_COMMAND)) = -0.4F;
  output(static_cast<int>(CostOutputIndex::PREVIOUS_STEER_RATE)) = -3.0F;
  const float second_cost = cost.computeComfortCost(control, output, 0);

  EXPECT_FLOAT_EQ(first_cost, second_cost);
}

TEST(FirstOrderDubinsBicycleSteeringCost, MatchingBoundaryHasZeroStepZeroSmoothnessCost)
{
  TestCost cost;
  TestCostParams params;
  params.steer_rate_l2_coeff = 1.0F;
  params.steer_accel_coeff = 1.0F;
  params.cmd_slew_coeff = 1.0F;
  cost.setParams(params);

  constexpr float kSteering = 0.23F;
  cost.setLastAppliedSteerCommand(kSteering);
  FirstOrderDubinsBicycle model;
  FirstOrderDubinsBicycle::state_array state = FirstOrderDubinsBicycle::state_array::Zero();
  FirstOrderDubinsBicycle::state_array next_state = FirstOrderDubinsBicycle::state_array::Zero();
  FirstOrderDubinsBicycle::state_array state_der = FirstOrderDubinsBicycle::state_array::Zero();
  TestCost::control_array control = TestCost::control_array::Zero();
  TestCost::output_array output = TestCost::output_array::Zero();
  state(static_cast<int>(FirstOrderDubinsBicycleParams::StateIndex::STEER_ANGLE)) = kSteering;
  state(static_cast<int>(FirstOrderDubinsBicycleParams::StateIndex::LAST_STEER_COMMAND)) =
    kSteering;
  control(static_cast<int>(CostControlIndex::STEER_CMD)) = kSteering;
  model.step(state, next_state, state_der, control, output, 0.0F, 0.1F);

  const auto terms = cost.computeSteeringSmoothnessCost(control.data(), output.data(), 0);
  EXPECT_FLOAT_EQ(terms.steer_rate_l2_cost, 0.0F);
  EXPECT_FLOAT_EQ(terms.cmd_slew_cost, 0.0F);
  EXPECT_FLOAT_EQ(terms.steer_accel_cost, 0.0F);
}

TEST(FirstOrderDubinsBicycleSteeringCost, AlternatingCommandsCostMoreThanConstantCommands)
{
  TestCost cost;
  TestCostParams params;
  params.cmd_slew_coeff = 1.0F;
  cost.setParams(params);
  cost.setLastAppliedSteerCommand(0.1F);

  const auto accumulated_slew_cost = [&cost](const std::array<float, 3> & commands) {
    float total = 0.0F;
    TestCost::control_array control = TestCost::control_array::Zero();
    TestCost::output_array output = TestCost::output_array::Zero();
    for (size_t i = 0; i < commands.size(); ++i) {
      control(static_cast<int>(CostControlIndex::STEER_CMD)) = commands[i];
      if (i > 0U) {
        output(static_cast<int>(CostOutputIndex::PREVIOUS_STEER_COMMAND)) = commands[i - 1U];
      }
      total +=
        cost.computeSteeringSmoothnessCost(control.data(), output.data(), static_cast<int>(i))
          .cmd_slew_cost;
    }
    return total;
  };

  const float constant_cost = accumulated_slew_cost({0.1F, 0.1F, 0.1F});
  const float alternating_cost = accumulated_slew_cost({0.1F, -0.1F, 0.1F});
  EXPECT_FLOAT_EQ(constant_cost, 0.0F);
  EXPECT_GT(alternating_cost, constant_cost);
}

TEST(FirstOrderDubinsBicycleSteeringCost, UsesExecutedRateForQuadraticRateAndAcceleration)
{
  TestCost cost;
  TestCostParams params;
  params.steer_rate_l2_coeff = 2.0F;
  params.steer_accel_coeff = 3.0F;
  cost.setParams(params);

  TestCost::control_array control = TestCost::control_array::Zero();
  TestCost::output_array output = TestCost::output_array::Zero();
  output(static_cast<int>(CostOutputIndex::STEER_RATE)) = 1.5F;
  output(static_cast<int>(CostOutputIndex::PREVIOUS_STEER_RATE)) = 0.5F;

  const auto terms = cost.computeSteeringSmoothnessCost(control.data(), output.data(), 1);
  EXPECT_FLOAT_EQ(terms.steer_rate_l2_cost, 4.5F);
  EXPECT_NEAR(terms.steer_accel_cost, 300.0F, 1.0E-4F);
}

class TrajectoryValidatorTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    int device_count = 0;
    const cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "A CUDA device is required by the MPPI cost object";
    }
    cost_ = std::make_unique<TestCost>();
    cost_->GPUSetup();
  }

  void TearDown() override
  {
    if (cost_) {
      cost_->freeCudaMem();
    }
  }

  TestCostParams makeParams() const
  {
    TestCostParams params;
    params.boundary_threshold = 100.0F;
    params.ego_length = 0.825F;
    params.ego_width = 0.42F;
    params.ego_axle_to_box_center = 0.2F;
    params.obstacle_collision_margin = 0.0F;
    params.road_border_collision_margin = 0.0F;
    return params;
  }

  void setStraightReference()
  {
    std::array<float, kTestHorizon> x{};
    std::array<float, kTestHorizon> y{};
    std::array<float, kTestHorizon> velocity{};
    std::array<float, kTestHorizon> yaw{};
    for (int i = 0; i < kTestHorizon; ++i) {
      x[static_cast<std::size_t>(i)] = 0.2F * static_cast<float>(i + 1);
    }
    cost_->setReferenceTrajectory(x.data(), y.data(), velocity.data(), kTestHorizon, yaw.data());
  }

  detail::OptimizedState makeFirstPostStepState(const float y = 0.0F) const
  {
    detail::OptimizedState state;
    state.x = 0.2F;
    state.y = y;
    state.yaw = 0.0F;
    state.velocity = 2.0F;
    return state;
  }

  std::unique_ptr<TestCost> cost_;
};

TEST_F(TrajectoryValidatorTest, AppliesBoundaryThresholdSymmetricallyAndInclusively)
{
  auto params = makeParams();
  params.boundary_threshold = 0.5F;
  cost_->setParams(params);
  setStraightReference();

  struct Case
  {
    float lateral_offset;
    bool valid;
  };
  const std::vector<Case> cases = {{0.49F, true},  {0.5F, false},  {0.51F, false},
                                   {-0.49F, true}, {-0.5F, false}, {-0.51F, false}};

  for (const auto & test_case : cases) {
    const auto result = detail::validateOptimizedTrajectory(
      *cost_,
      std::vector<detail::OptimizedState>{makeFirstPostStepState(test_case.lateral_offset)});
    EXPECT_EQ(result.isValid(), test_case.valid) << "offset=" << test_case.lateral_offset;
    EXPECT_EQ(
      hasInvalidityReason(result.reasons, FirstOrderDubinsMppiInvalidityReason::lateral_boundary),
      !test_case.valid)
      << "offset=" << test_case.lateral_offset;
  }
}

TEST_F(TrajectoryValidatorTest, IgnoresLongitudinalLagOnCurvesWhenCheckingLateralBoundary)
{
  auto params = makeParams();
  params.boundary_threshold = 0.1F;
  cost_->setParams(params);

  constexpr float kPi = 3.14159265358979323846F;
  constexpr float kRadius = 2.0F;
  constexpr int kArcEndIndex = 6;
  std::array<float, kTestHorizon> reference_x{};
  std::array<float, kTestHorizon> reference_y{};
  std::array<float, kTestHorizon> reference_velocity{};
  std::array<float, kTestHorizon> reference_yaw{};
  for (int i = 0; i < kTestHorizon; ++i) {
    const int arc_index = std::min(i, kArcEndIndex);
    const float angle = static_cast<float>(arc_index) * kPi / 12.0F;
    reference_x[static_cast<std::size_t>(i)] = kRadius * std::sin(angle);
    reference_y[static_cast<std::size_t>(i)] = kRadius * (1.0F - std::cos(angle));
    reference_yaw[static_cast<std::size_t>(i)] = angle;
  }
  cost_->setReferenceTrajectory(
    reference_x.data(), reference_y.data(), reference_velocity.data(), kTestHorizon,
    reference_yaw.data());

  std::vector<detail::OptimizedState> states;
  states.reserve(static_cast<std::size_t>(kArcEndIndex + 1));
  for (int i = 0; i <= kArcEndIndex; ++i) {
    detail::OptimizedState state;
    state.x = reference_x[static_cast<std::size_t>(i)];
    state.y = reference_y[static_cast<std::size_t>(i)];
    state.yaw = reference_yaw[static_cast<std::size_t>(i)];
    states.push_back(state);
  }

  constexpr int kCheckedTimestep = 6;
  constexpr int kLaggedReferenceIndex = kCheckedTimestep - 2;
  auto & lagged_state = states[static_cast<std::size_t>(kCheckedTimestep)];
  lagged_state.x = reference_x[static_cast<std::size_t>(kLaggedReferenceIndex)];
  lagged_state.y = reference_y[static_cast<std::size_t>(kLaggedReferenceIndex)];
  lagged_state.yaw = reference_yaw[static_cast<std::size_t>(kLaggedReferenceIndex)];

  EXPECT_NEAR(
    cost_->computeLocalLateralDistanceValue(lagged_state.x, lagged_state.y, kCheckedTimestep), 0.0F,
    1.0E-6F);
  EXPECT_FALSE(cost_->exceedsLateralBoundary(lagged_state.x, lagged_state.y, kCheckedTimestep));
  EXPECT_TRUE(detail::validateOptimizedTrajectory(*cost_, states).isValid());

  constexpr int kSegmentIndex = 4;
  const float segment_dx = reference_x[kSegmentIndex + 1] - reference_x[kSegmentIndex];
  const float segment_dy = reference_y[kSegmentIndex + 1] - reference_y[kSegmentIndex];
  const float segment_length = std::hypot(segment_dx, segment_dy);
  const float midpoint_x = 0.5F * (reference_x[kSegmentIndex] + reference_x[kSegmentIndex + 1]);
  const float midpoint_y = 0.5F * (reference_y[kSegmentIndex] + reference_y[kSegmentIndex + 1]);
  const float outward_normal_x = segment_dy / segment_length;
  const float outward_normal_y = -segment_dx / segment_length;
  EXPECT_FALSE(cost_->exceedsLateralBoundary(
    midpoint_x + 0.08F * outward_normal_x, midpoint_y + 0.08F * outward_normal_y, kSegmentIndex));
  EXPECT_TRUE(cost_->exceedsLateralBoundary(
    midpoint_x + 0.12F * outward_normal_x, midpoint_y + 0.12F * outward_normal_y, kSegmentIndex));
}

TEST_F(TrajectoryValidatorTest, UsesNearestPointOnSegmentRatherThanNearestSample)
{
  auto params = makeParams();
  params.boundary_threshold = 0.01F;
  cost_->setParams(params);

  std::array<float, kTestHorizon> reference_x{};
  std::array<float, kTestHorizon> reference_y{};
  std::array<float, kTestHorizon> reference_velocity{};
  std::array<float, kTestHorizon> reference_yaw{};
  reference_x[0] = 0.0F;
  reference_y[0] = 0.0F;
  for (int i = 1; i < kTestHorizon; ++i) {
    reference_x[static_cast<std::size_t>(i)] = 2.0F;
    reference_y[static_cast<std::size_t>(i)] = 2.0F;
  }
  cost_->setReferenceTrajectory(
    reference_x.data(), reference_y.data(), reference_velocity.data(), kTestHorizon,
    reference_yaw.data());

  EXPECT_NEAR(cost_->computeLocalLateralDistanceValue(1.0F, 1.0F, 0), 0.0F, 1.0E-6F);
  EXPECT_FALSE(cost_->exceedsLateralBoundary(1.0F, 1.0F, 0));
}

TEST_F(TrajectoryValidatorTest, DoesNotMatchTemporallyDistantParallelBranch)
{
  auto params = makeParams();
  params.boundary_threshold = 0.5F;
  cost_->setParams(params);

  std::array<float, kTestHorizon> reference_x{};
  std::array<float, kTestHorizon> reference_y{};
  std::array<float, kTestHorizon> reference_velocity{};
  std::array<float, kTestHorizon> reference_yaw{};
  for (int i = 0; i < kTestHorizon / 2; ++i) {
    reference_x[static_cast<std::size_t>(i)] = 0.2F * static_cast<float>(i);
    reference_y[static_cast<std::size_t>(i)] = 0.0F;
  }
  for (int i = kTestHorizon / 2; i < kTestHorizon; ++i) {
    reference_x[static_cast<std::size_t>(i)] = 0.2F * static_cast<float>(kTestHorizon - 1 - i);
    reference_y[static_cast<std::size_t>(i)] = 1.0F;
    reference_yaw[static_cast<std::size_t>(i)] = 3.14159265358979323846F;
  }
  cost_->setReferenceTrajectory(
    reference_x.data(), reference_y.data(), reference_velocity.data(), kTestHorizon,
    reference_yaw.data());

  constexpr float kStateX = 2.0F;
  constexpr float kStateY = 1.0F;
  constexpr int kTimestep = 10;
  EXPECT_NEAR(cost_->computeLateralDistanceValue(kStateX, kStateY), 0.0F, 1.0E-6F);
  EXPECT_NEAR(cost_->computeLocalLateralDistanceValue(kStateX, kStateY, kTimestep), 1.0F, 1.0E-6F);
  EXPECT_TRUE(cost_->exceedsLateralBoundary(kStateX, kStateY, kTimestep));
}

TEST_F(TrajectoryValidatorTest, HandlesDuplicatedReferencePoints)
{
  auto params = makeParams();
  params.boundary_threshold = 0.01F;
  cost_->setParams(params);

  std::array<float, kTestHorizon> reference_x{};
  std::array<float, kTestHorizon> reference_y{};
  std::array<float, kTestHorizon> reference_velocity{};
  std::array<float, kTestHorizon> reference_yaw{};
  reference_x[0] = 0.0F;
  reference_x[1] = 0.0F;
  for (int i = 2; i < kTestHorizon; ++i) {
    reference_x[static_cast<std::size_t>(i)] = 1.0F;
  }
  cost_->setReferenceTrajectory(
    reference_x.data(), reference_y.data(), reference_velocity.data(), kTestHorizon,
    reference_yaw.data());

  const float distance = cost_->computeLocalLateralDistanceValue(0.5F, 0.0F, 1);
  EXPECT_TRUE(std::isfinite(distance));
  EXPECT_NEAR(distance, 0.0F, 1.0E-6F);
  EXPECT_FALSE(cost_->exceedsLateralBoundary(0.5F, 0.0F, 1));
}

TEST_F(TrajectoryValidatorTest, RoadBorderMarginInflatesTheEgoFootprint)
{
  auto params = makeParams();
  cost_->setParams(params);
  setStraightReference();
  cost_->setRoadBorderSegments({Segment{-1.0F, 0.31F, 2.0F, 0.31F}});
  const std::vector<detail::OptimizedState> states{makeFirstPostStepState()};

  const auto without_margin = detail::validateOptimizedTrajectory(*cost_, states);
  EXPECT_TRUE(without_margin.isValid());

  params.road_border_collision_margin = 0.2F;
  cost_->setParams(params);
  const auto with_margin = detail::validateOptimizedTrajectory(*cost_, states);
  EXPECT_FALSE(with_margin.isValid());
  EXPECT_TRUE(
    hasInvalidityReason(with_margin.reasons, FirstOrderDubinsMppiInvalidityReason::road_border));
  ASSERT_TRUE(with_margin.first_invalid_index.has_value());
  EXPECT_EQ(with_margin.first_invalid_index.value(), 0U);
}

TEST_F(TrajectoryValidatorTest, ObstacleMarginInflatesTheEgoOrientedBox)
{
  auto params = makeParams();
  cost_->setParams(params);
  setStraightReference();
  constexpr float obstacle_x = 0.4F;
  constexpr float obstacle_y = 0.41F;
  constexpr float obstacle_yaw = 0.0F;
  constexpr float obstacle_half_length = 0.1F;
  constexpr float obstacle_half_width = 0.1F;
  cost_->setOrientedBoxObstacles(
    &obstacle_x, &obstacle_y, &obstacle_yaw, &obstacle_half_length, &obstacle_half_width, 1);
  const std::vector<detail::OptimizedState> states{makeFirstPostStepState()};

  const auto without_margin = detail::validateOptimizedTrajectory(*cost_, states);
  EXPECT_TRUE(without_margin.isValid());

  params.obstacle_collision_margin = 0.2F;
  cost_->setParams(params);
  const auto with_margin = detail::validateOptimizedTrajectory(*cost_, states);
  EXPECT_FALSE(with_margin.isValid());
  EXPECT_TRUE(
    hasInvalidityReason(with_margin.reasons, FirstOrderDubinsMppiInvalidityReason::obstacle));
  ASSERT_TRUE(with_margin.first_invalid_index.has_value());
  EXPECT_EQ(with_margin.first_invalid_index.value(), 0U);
}

TEST_F(TrajectoryValidatorTest, DetectsLateralBoundaryViolationsAcrossHorizonAndProfiles)
{
  auto params = makeParams();
  params.boundary_threshold = 0.50F;
  cost_->setParams(params);
  setStraightReference();  // Reference trajectory is along y = 0.0

  struct DeviationCase
  {
    std::string name;
    std::vector<float> y_profile;
    bool expected_valid;
    std::optional<std::size_t> expected_first_invalid_idx;
  };

  std::vector<DeviationCase> test_cases;

  // Case 1: Strictly within threshold across the entire horizon -> Valid
  test_cases.push_back({"all_valid", std::vector<float>(kTestHorizon, 0.40F), true, std::nullopt});

  // Case 2: Constant violation from step 0 (Left and Right) -> Invalid at index 0
  test_cases.push_back(
    {"invalid_at_start_right", std::vector<float>(kTestHorizon, 0.60F), false, 0U});
  test_cases.push_back(
    {"invalid_at_start_left", std::vector<float>(kTestHorizon, -0.60F), false, 0U});

  // Case 3: Progressive drift is below 0.50F at step 5 and breaches it at step 6.
  {
    std::vector<float> drift(kTestHorizon, 0.0F);
    for (int i = 0; i < kTestHorizon; ++i) {
      drift[static_cast<std::size_t>(i)] =
        0.09F * static_cast<float>(i);  // 0.45F at i=5; 0.54F at i=6.
    }
    test_cases.push_back({"progressive_drift_mid_horizon", drift, false, 6U});
  }

  // Case 4: Single-step spike at the tail end of the horizon (step 79)
  {
    std::vector<float> tail_spike(kTestHorizon, 0.0F);
    tail_spike.back() = 0.55F;
    test_cases.push_back(
      {"tail_step_violation", tail_spike, false, static_cast<std::size_t>(kTestHorizon - 1)});
  }

  // Case 5: Exact boundary edge (0.50F is inclusive rejection: offset >= threshold)
  test_cases.push_back(
    {"exact_threshold_edge", std::vector<float>(kTestHorizon, 0.50F), false, 0U});

  for (const auto & tc : test_cases) {
    std::vector<detail::OptimizedState> states(kTestHorizon);
    for (std::size_t i = 0; i < static_cast<std::size_t>(kTestHorizon); ++i) {
      states[i].x = 0.20F * static_cast<float>(i + 1U);
      states[i].y = tc.y_profile[i];
      states[i].yaw = 0.0F;
      states[i].velocity = 2.0F;
    }
    const auto result = detail::validateOptimizedTrajectory(*cost_, states);
    EXPECT_EQ(result.isValid(), tc.expected_valid) << "Failed case: " << tc.name;
    if (!tc.expected_valid) {
      EXPECT_TRUE(
        hasInvalidityReason(result.reasons, FirstOrderDubinsMppiInvalidityReason::lateral_boundary))
        << "Failed case: " << tc.name;
      ASSERT_TRUE(result.first_invalid_index.has_value()) << "Failed case: " << tc.name;
      EXPECT_EQ(result.first_invalid_index.value(), tc.expected_first_invalid_idx.value())
        << "Failed case: " << tc.name;
    }
  }
}

}  // namespace
}  // namespace autoware::mppi_optimizer
