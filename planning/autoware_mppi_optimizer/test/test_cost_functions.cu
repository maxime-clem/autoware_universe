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

#include <mppi/cost_functions/dubins/first_order_dubins_bicycle_cost.cuh>
#include <mppi/dynamics/dubins/first_order_dubins_bicycle.cuh>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace autoware::mppi_optimizer
{
namespace
{

constexpr int NUM_TIMESTEPS_TEST = detail::kMppiHorizon;

using Cost = FirstOrderDubinsBicycleCost<NUM_TIMESTEPS_TEST>;
using CostParams = FirstOrderDubinsBicycleCostParams<NUM_TIMESTEPS_TEST>;
using OutputIndex = FirstOrderDubinsBicycleParams::OutputIndex;
using ControlIndex = FirstOrderDubinsBicycleParams::ControlIndex;

using output_array = Cost::output_array;
using control_array = Cost::control_array;

// ============================================================================
// TEST FIXTURE
// ============================================================================
class CostFunctionTest : public ::testing::Test
{
protected:
  std::unique_ptr<Cost> cost_impl_;
  CostParams params_;

  // Base states to pass to the evaluator
  output_array base_y_;
  control_array base_u_;

  void SetUp() override
  {
    int device_count = 0;
    const cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "A CUDA device is required by the MPPI cost object";
    }

    cost_impl_ = std::make_unique<Cost>();
    cost_impl_->GPUSetup();

    params_.desired_speed = 5.0F;
    params_.boundary_threshold = 2.0F;
    params_.ego_axle_to_box_center = 1.0F;
    params_.obstacle_collision_margin = 0.5F;
    params_.ego_length = 4.0F;
    params_.ego_width = 2.0F;

    // Cost coefficients (normalized to 1.0 for testing so raw values pass through)
    params_.speed_coeff = 1.0F;
    params_.track_coeff = 1.0F;
    params_.heading_coeff = 1.0F;
    params_.lateral_distance_coeff = 1.0F;
    params_.lateral_yaw_error_coeff = 1.0F;
    params_.track_center_coeff = 1.0F;
    params_.drivable_area_crossing_coeff = 1000.0F;
    params_.accel_cmd_coeff = 1.0F;
    params_.steer_cmd_coeff = 1.0F;
    params_.lateral_acceleration_coeff = 1.0F;
    params_.lateral_jerk_coeff = 1.0F;
    params_.longitudinal_jerk_coeff = 1.0F;
    params_.steer_rate_coeff = 1.0F;
    params_.corner_buffer_coeff = 1.0F;
    params_.corner_safe_margin = 0.5F;
    params_.crash_coeff = 10000.0F;

    params_.accel_time_constant = 0.1F;
    params_.steer_time_constant = 0.1F;
    params_.wheel_base = 2.5F;
    params_.track_terminal_scale = 10.0F;

    cost_impl_->setParams(params_);

    // Setup standard reference trajectory (Straight line along X axis)
    std::vector<float> ref_x(NUM_TIMESTEPS_TEST);
    std::vector<float> ref_y(NUM_TIMESTEPS_TEST, 0.0F);
    std::vector<float> ref_yaw(NUM_TIMESTEPS_TEST, 0.0F);
    std::vector<float> ref_v(NUM_TIMESTEPS_TEST, params_.desired_speed);

    for (int i = 0; i < NUM_TIMESTEPS_TEST; ++i) {
      ref_x[i] = static_cast<float>(i);  // 1 meter per timestep
    }

    cost_impl_->setReferenceTrajectory(
      ref_x.data(), ref_y.data(), ref_v.data(), NUM_TIMESTEPS_TEST, ref_yaw.data());

    base_y_ = output_array::Zero();
    base_u_ = control_array::Zero();

    // Set base speed to desired speed so speed cost is 0 by default
    base_y_[static_cast<int>(OutputIndex::TOTAL_VELOCITY)] = params_.desired_speed;
    base_y_[static_cast<int>(OutputIndex::BASELINK_VEL_B_X)] = params_.desired_speed;
  }

  void TearDown() override
  {
    if (cost_impl_) {
      cost_impl_->freeCudaMem();
    }
  }

  // HELPER: Evaluates the cost breakdown at a specific timestep given the state
  FirstOrderDubinsMppiCostBreakdown evaluateState(
    const output_array & y, const control_array & u, int timestep = 0)
  {
    int crash_status = 0;
    return cost_impl_->computeRunningCostBreakdown(y, u, timestep, &crash_status);
  }
};

// ============================================================================
// TEST CASES
// ============================================================================

// 1. Monotonicity Test: Lateral Distance
// Sweeping the vehicle away from the reference line (Y=0) should strictly increase the lateral costs.
TEST_F(CostFunctionTest, LateralCostsAreMonotonic)
{
  float prev_track_cost = -1.0F;
  float prev_lateral_dist_cost = -1.0F;

  // Sweep within boundary_threshold (2.0m) to evaluate purely lateral tracking costs
  for (float y_offset = 0.0F; y_offset <= 1.5F; y_offset += 0.25F) {
    output_array state = base_y_;
    state[static_cast<int>(OutputIndex::BASELINK_POS_I_X)] = 0.0F;
    state[static_cast<int>(OutputIndex::BASELINK_POS_I_Y)] = y_offset;

    auto breakdown = evaluateState(state, base_u_);

    if (y_offset > 0.0F) {
      // Expect costs to strictly increase as we move further from the path
      EXPECT_GT(breakdown.track, prev_track_cost)
        << "Track cost failed to increase monotonically at y = " << y_offset;
      EXPECT_GT(breakdown.lateral_distance, prev_lateral_dist_cost)
        << "Lateral distance cost failed to increase monotonically at y = " << y_offset;
    }

    prev_track_cost = breakdown.track;
    prev_lateral_dist_cost = breakdown.lateral_distance;
  }
}

// 2. Monotonicity Test: Heading Error
// Sweeping the yaw away from the reference yaw (0.0) should strictly increase heading costs.
TEST_F(CostFunctionTest, HeadingCostIsMonotonic)
{
  float prev_heading_cost = -1.0F;

  // Sweep from 0 to Pi (180 degrees)
  for (float yaw = 0.0F; yaw <= static_cast<float>(M_PI); yaw += 0.1F) {
    output_array state = base_y_;
    state[static_cast<int>(OutputIndex::BASELINK_POS_I_X)] = 0.0F;
    state[static_cast<int>(OutputIndex::BASELINK_POS_I_Y)] = 0.0F;
    state[static_cast<int>(OutputIndex::YAW)] = yaw;

    auto breakdown = evaluateState(state, base_u_);

    if (yaw > 0.0F) {
      EXPECT_GT(breakdown.heading, prev_heading_cost)
        << "Heading cost failed to increase monotonically at yaw = " << yaw;
    }

    prev_heading_cost = breakdown.heading;
  }
}

// 3. Monotonicity Test: Speed Error
// Sweeping velocity away from the desired velocity should strictly increase speed cost.
TEST_F(CostFunctionTest, SpeedCostIsMonotonic)
{
  float prev_speed_cost = -1.0F;
  const float desired_v = params_.desired_speed;

  for (float v = desired_v; v >= 0.0F; v -= 0.5F) {
    output_array state = base_y_;
    state[static_cast<int>(OutputIndex::TOTAL_VELOCITY)] = v;

    auto breakdown = evaluateState(state, base_u_);

    if (v < desired_v) {
      EXPECT_GT(breakdown.speed, prev_speed_cost)
        << "Speed cost failed to increase as velocity dropped to " << v;
    }

    prev_speed_cost = breakdown.speed;
  }
}

// 4. Smoothness / Continuity Test: Closest Segment Tangent
// Tests `computeLateralYawErrorValue` to ensure moving along a corner doesn't cause jagged cost spikes.
TEST_F(CostFunctionTest, LateralYawErrorIsContinuousAroundCorners)
{
  // Create a 90-degree corner reference trajectory
  std::vector<float> ref_x(NUM_TIMESTEPS_TEST);
  std::vector<float> ref_y(NUM_TIMESTEPS_TEST);

  for (int i = 0; i < NUM_TIMESTEPS_TEST; ++i) {
    if (i < NUM_TIMESTEPS_TEST / 2) {
      ref_x[i] = static_cast<float>(i);
      ref_y[i] = 0.0F;
    } else {
      ref_x[i] = static_cast<float>(NUM_TIMESTEPS_TEST / 2 - 1);
      ref_y[i] = static_cast<float>(i - NUM_TIMESTEPS_TEST / 2);
    }
  }
  cost_impl_->setReferenceTrajectory(
    ref_x.data(), ref_y.data(), nullptr, NUM_TIMESTEPS_TEST, nullptr);

  // Sweep ego vehicle in a smooth arc around the corner
  float prev_lat_yaw_cost = -1.0F;
  float max_cost_delta = 0.0F;

  for (float t = 0.0F; t <= static_cast<float>(M_PI) / 2.0F; t += 0.05F) {
    output_array state = base_y_;
    // Arc radius of 2.0 meters around the pivot point
    const float pivot_x = static_cast<float>(NUM_TIMESTEPS_TEST / 2 - 1);
    const float pivot_y = 0.0F;

    state[static_cast<int>(OutputIndex::BASELINK_POS_I_X)] =
      pivot_x + 2.0F * std::cos(t - static_cast<float>(M_PI) / 2.0F);
    state[static_cast<int>(OutputIndex::BASELINK_POS_I_Y)] =
      pivot_y + 2.0F * std::sin(t - static_cast<float>(M_PI) / 2.0F);
    state[static_cast<int>(OutputIndex::YAW)] = t;

    auto breakdown = evaluateState(state, base_u_);

    if (t > 0.0F) {
      const float cost_delta = std::abs(breakdown.lateral_yaw_error - prev_lat_yaw_cost);
      max_cost_delta = std::max(max_cost_delta, cost_delta);

      // If the delta is massive (e.g., jumps from 0.1 to 1.5 in a 0.05rad step),
      // the closest-segment logic is creating an artificial discontinuity.
      EXPECT_LT(cost_delta, 0.5F)
        << "Discontinuity detected in lateral yaw error cost at arc angle " << t;
    }
    prev_lat_yaw_cost = breakdown.lateral_yaw_error;
  }
}

// 5. Crash Latching Logic Test
// Ensures that once a crash is detected, the cost heavily penalizes the state
TEST_F(CostFunctionTest, CrashCostLatchesCorrectly)
{
  output_array state = base_y_;

  // 1. Safe state
  int crash_status = 0;
  auto safe_breakdown = cost_impl_->computeRunningCostBreakdown(state, base_u_, 0, &crash_status);
  EXPECT_EQ(crash_status, 0);
  EXPECT_EQ(safe_breakdown.crash, 0.0F);

  // 2. Trigger a lateral boundary exceedance
  state[static_cast<int>(OutputIndex::BASELINK_POS_I_Y)] = params_.boundary_threshold + 0.1F;
  auto crash_breakdown = cost_impl_->computeRunningCostBreakdown(state, base_u_, 0, &crash_status);

  EXPECT_GT(crash_status, 0) << "Crash status should be positive for boundary violation";
  EXPECT_GT(crash_breakdown.crash, 0.0F);

  // 3. Move back to safe state, but pass the latched crash_status
  state[static_cast<int>(OutputIndex::BASELINK_POS_I_Y)] = 0.0F;
  auto latched_breakdown = cost_impl_->computeRunningCostBreakdown(state, base_u_, 0, &crash_status);

  EXPECT_GT(latched_breakdown.crash, 0.0F)
    << "Crash cost failed to latch despite safe current state";
}

// 6. Monotonicity Test: Comfort Costs (Lateral Acceleration)
// Sweeping the steering angle / lateral acceleration should strictly increase lateral acceleration cost.
TEST_F(CostFunctionTest, ComfortCostLateralAccelerationIsMonotonic)
{
  float prev_lat_accel_cost = -1.0F;

  // Sweep steering angle from 0.0 to 0.4 rad with non-zero forward speed
  for (float steer = 0.0F; steer <= 0.40F; steer += 0.05F) {
    output_array state = base_y_;
    state[static_cast<int>(OutputIndex::BASELINK_VEL_B_X)] = params_.desired_speed;
    state[static_cast<int>(OutputIndex::TOTAL_VELOCITY)] = params_.desired_speed;
    state[static_cast<int>(OutputIndex::STEER_ANGLE)] = steer;

    auto breakdown = evaluateState(state, base_u_);

    if (steer > 0.0F) {
      EXPECT_GT(breakdown.lateral_acceleration, prev_lat_accel_cost)
        << "Lateral acceleration cost failed to increase monotonically at steer = " << steer;
    }

    prev_lat_accel_cost = breakdown.lateral_acceleration;
  }
}

// 7. Boundary Test: Drivable Area Crossing
// Ensures the massive penalty applies exactly when the vehicle's bounding box crosses the boundary.
TEST_F(CostFunctionTest, DrivableAreaCrossingAppliesBoundaryPenalty)
{
  // Boundary line parallel to X-axis at Y = 3.0 meters
  constexpr float boundary_y = 3.0F;
  cost_impl_->setDrivableAreaSegments({Segment{-10.0F, boundary_y, 60.0F, boundary_y}});

  // Ego vehicle with width = 2.0 (half-width = 1.0) and yaw = 0.0
  // Left edge is at y + 1.0 m. Boundary crossing occurs when y + 1.0 >= 3.0, i.e., y >= 2.0 m.
  const float half_width = params_.ego_width * 0.5F;
  const float crossing_threshold_y = boundary_y - half_width;  // 2.0 m

  // Inside the boundary (clearance margin): drivable area cost should be exactly 0.0
  output_array safe_state = base_y_;
  safe_state[static_cast<int>(OutputIndex::BASELINK_POS_I_X)] = 5.0F;
  safe_state[static_cast<int>(OutputIndex::BASELINK_POS_I_Y)] = crossing_threshold_y - 0.10F;  // 1.90 m
  safe_state[static_cast<int>(OutputIndex::YAW)] = 0.0F;
  auto safe_breakdown = evaluateState(safe_state, base_u_);
  EXPECT_FLOAT_EQ(safe_breakdown.drivable_area, 0.0F)
    << "Expected zero drivable area penalty inside the drivable area";

  // Just inside the boundary edge (1.99 m): still 0.0 penalty
  safe_state[static_cast<int>(OutputIndex::BASELINK_POS_I_Y)] = crossing_threshold_y - 0.01F;  // 1.99 m
  auto near_edge_breakdown = evaluateState(safe_state, base_u_);
  EXPECT_FLOAT_EQ(near_edge_breakdown.drivable_area, 0.0F)
    << "Expected zero penalty before vehicle bounding box crosses the boundary";

  // Crossing the boundary (2.01 m): massive penalty applied
  output_array crossed_state = base_y_;
  crossed_state[static_cast<int>(OutputIndex::BASELINK_POS_I_X)] = 5.0F;
  crossed_state[static_cast<int>(OutputIndex::BASELINK_POS_I_Y)] = crossing_threshold_y + 0.01F;  // 2.01 m
  crossed_state[static_cast<int>(OutputIndex::YAW)] = 0.0F;
  auto crossed_breakdown = evaluateState(crossed_state, base_u_);
  EXPECT_FLOAT_EQ(crossed_breakdown.drivable_area, params_.drivable_area_crossing_coeff)
    << "Expected full crossing penalty when vehicle bounding box intersects the boundary";

  // Further outside the boundary (2.50 m)
  crossed_state[static_cast<int>(OutputIndex::BASELINK_POS_I_Y)] = crossing_threshold_y + 0.50F;  // 2.50 m
  auto far_crossed_breakdown = evaluateState(crossed_state, base_u_);
  EXPECT_FLOAT_EQ(far_crossed_breakdown.drivable_area, params_.drivable_area_crossing_coeff);

  cost_impl_->clearDrivableAreaSegments();
}

// 8. Terminal Cost: Track Terminal Scale Multiplier
// Verifies that the terminal cost applies the track_terminal_scale multiplier correctly.
TEST_F(CostFunctionTest, TerminalCostAppliesTrackTerminalScale)
{
  params_.track_terminal_scale = 10.0F;
  cost_impl_->setParams(params_);

  output_array terminal_state = base_y_;
  // Set terminal state with lateral displacement from reference at final timestep
  terminal_state[static_cast<int>(OutputIndex::BASELINK_POS_I_X)] =
    static_cast<float>(NUM_TIMESTEPS_TEST - 1);
  terminal_state[static_cast<int>(OutputIndex::BASELINK_POS_I_Y)] = 1.5F;

  int running_crash_status = 0;
  auto running_breakdown = cost_impl_->computeRunningCostBreakdown(
    terminal_state, base_u_, NUM_TIMESTEPS_TEST - 1, &running_crash_status);
  auto terminal_breakdown = cost_impl_->computeTerminalCostBreakdown(terminal_state);

  EXPECT_GT(running_breakdown.track, 0.0F);
  EXPECT_NEAR(
    terminal_breakdown.track, running_breakdown.track * params_.track_terminal_scale, 1.0E-5F)
    << "Terminal track cost should equal running track cost multiplied by track_terminal_scale";

  // Test with a different scale factor
  params_.track_terminal_scale = 3.5F;
  cost_impl_->setParams(params_);

  auto scaled_terminal = cost_impl_->computeTerminalCostBreakdown(terminal_state);
  EXPECT_NEAR(
    scaled_terminal.track, running_breakdown.track * 3.5F, 1.0E-5F)
    << "Terminal track cost should scale proportionally with modified track_terminal_scale";
}

// 9. Cost Trade-off & Weighting Test: Speed vs Comfort
// Proves the optimizer correctly balances competing costs: maintaining speed through a curve
// versus slowing down to reduce lateral acceleration.
TEST_F(CostFunctionTest, SpeedVsComfortTradeoff)
{
  // Setup a reference trajectory with a curved section
  std::vector<float> ref_x(NUM_TIMESTEPS_TEST);
  std::vector<float> ref_y(NUM_TIMESTEPS_TEST);
  std::vector<float> ref_yaw(NUM_TIMESTEPS_TEST);
  std::vector<float> ref_v(NUM_TIMESTEPS_TEST, params_.desired_speed);

  for (int i = 0; i < NUM_TIMESTEPS_TEST; ++i) {
    const float t = static_cast<float>(i) * 0.1F;
    ref_x[i] = 10.0F * std::sin(t);
    ref_y[i] = 10.0F * (1.0F - std::cos(t));
    ref_yaw[i] = t;
  }
  cost_impl_->setReferenceTrajectory(
    ref_x.data(), ref_y.data(), ref_v.data(), NUM_TIMESTEPS_TEST, ref_yaw.data());

  // State A: Maintains desired speed (low speed cost) but high steering angle (high comfort cost)
  output_array state_a = base_y_;
  state_a[static_cast<int>(OutputIndex::BASELINK_POS_I_X)] = ref_x[5];
  state_a[static_cast<int>(OutputIndex::BASELINK_POS_I_Y)] = ref_y[5];
  state_a[static_cast<int>(OutputIndex::YAW)] = ref_yaw[5];
  state_a[static_cast<int>(OutputIndex::TOTAL_VELOCITY)] = params_.desired_speed;
  state_a[static_cast<int>(OutputIndex::BASELINK_VEL_B_X)] = params_.desired_speed;
  state_a[static_cast<int>(OutputIndex::STEER_ANGLE)] = 0.35F;

  control_array u_a = base_u_;
  u_a[static_cast<int>(ControlIndex::STEER_CMD)] = 0.35F; // Match steer to prevent lateral jerk

  // State B: Slows down significantly (high speed cost) but minimal steering angle (low comfort cost)
  output_array state_b = base_y_;
  state_b[static_cast<int>(OutputIndex::BASELINK_POS_I_X)] = ref_x[5];
  state_b[static_cast<int>(OutputIndex::BASELINK_POS_I_Y)] = ref_y[5];
  state_b[static_cast<int>(OutputIndex::YAW)] = ref_yaw[5];
  state_b[static_cast<int>(OutputIndex::TOTAL_VELOCITY)] = 1.0F;
  state_b[static_cast<int>(OutputIndex::BASELINK_VEL_B_X)] = 1.0F;
  state_b[static_cast<int>(OutputIndex::STEER_ANGLE)] = 0.05F;

  control_array u_b = base_u_;
  u_b[static_cast<int>(ControlIndex::STEER_CMD)] = 0.05F; // Match steer to prevent lateral jerk

  // Case 1: High speed_coeff, Low lateral_acceleration_coeff -> State A should win (lower total cost)
  params_.speed_coeff = 20.0F;
  params_.lateral_acceleration_coeff = 0.01F;
  cost_impl_->setParams(params_);

  auto breakdown_a_speed_priority = evaluateState(state_a, u_a, 5);
  auto breakdown_b_speed_priority = evaluateState(state_b, u_b, 5);
  EXPECT_LT(breakdown_a_speed_priority.total, breakdown_b_speed_priority.total)
    << "State A (maintaining speed) should have lower cost when speed weight dominates";

  // Case 2: Low speed_coeff, High lateral_acceleration_coeff -> State B should win (lower total cost)
  params_.speed_coeff = 0.01F;
  params_.lateral_acceleration_coeff = 20.0F;
  cost_impl_->setParams(params_);

  auto breakdown_a_comfort_priority = evaluateState(state_a, u_a, 5);
  auto breakdown_b_comfort_priority = evaluateState(state_b, u_b, 5);
  EXPECT_LT(breakdown_b_comfort_priority.total, breakdown_a_comfort_priority.total)
    << "State B (slowing down for comfort) should have lower cost when comfort weight dominates";
}

// 10. Singularity & Robustness Test: Zero Velocity
// Ensures that evaluateState does not return NaN or Inf when velocity is zero.
TEST_F(CostFunctionTest, HandlesZeroVelocity)
{
  output_array zero_vel_state = base_y_;
  zero_vel_state[static_cast<int>(OutputIndex::TOTAL_VELOCITY)] = 0.0F;
  zero_vel_state[static_cast<int>(OutputIndex::BASELINK_VEL_B_X)] = 0.0F;
  zero_vel_state[static_cast<int>(OutputIndex::BASELINK_VEL_B_Y)] = 0.0F;
  zero_vel_state[static_cast<int>(OutputIndex::STEER_ANGLE)] = 0.2F;
  zero_vel_state[static_cast<int>(OutputIndex::ACCELERATION)] = 0.0F;

  auto breakdown = evaluateState(zero_vel_state, base_u_, 0);

  // Assert no field produces NaN or Infinity
  EXPECT_FALSE(std::isnan(breakdown.speed));
  EXPECT_FALSE(std::isinf(breakdown.speed));
  EXPECT_FALSE(std::isnan(breakdown.track));
  EXPECT_FALSE(std::isinf(breakdown.track));
  EXPECT_FALSE(std::isnan(breakdown.heading));
  EXPECT_FALSE(std::isinf(breakdown.heading));
  EXPECT_FALSE(std::isnan(breakdown.lateral_distance));
  EXPECT_FALSE(std::isinf(breakdown.lateral_distance));
  EXPECT_FALSE(std::isnan(breakdown.lateral_yaw_error));
  EXPECT_FALSE(std::isinf(breakdown.lateral_yaw_error));
  EXPECT_FALSE(std::isnan(breakdown.lateral_acceleration));
  EXPECT_FALSE(std::isinf(breakdown.lateral_acceleration));
  EXPECT_FALSE(std::isnan(breakdown.lateral_jerk));
  EXPECT_FALSE(std::isinf(breakdown.lateral_jerk));
  EXPECT_FALSE(std::isnan(breakdown.longitudinal_jerk));
  EXPECT_FALSE(std::isinf(breakdown.longitudinal_jerk));
  EXPECT_FALSE(std::isnan(breakdown.steering_rate));
  EXPECT_FALSE(std::isinf(breakdown.steering_rate));
  EXPECT_FALSE(std::isnan(breakdown.total));
  EXPECT_FALSE(std::isinf(breakdown.total));

  // At zero velocity, lateral acceleration must be zero (a_y = v^2 * kappa = 0)
  EXPECT_FLOAT_EQ(breakdown.lateral_acceleration, 0.0F);
  EXPECT_GT(breakdown.speed, 0.0F)
    << "Speed error cost should be positive when stopped away from desired speed";
}

// 11. Edge Case Test: Extreme Yaw Wrapping
// Verifies that 180-degree offset and large wrapped angles do not produce NaNs, Infs, or unbounded costs.
TEST_F(CostFunctionTest, HandlesExtremeYawWrapping)
{
  output_array state = base_y_;
  state[static_cast<int>(OutputIndex::BASELINK_POS_I_X)] = 0.0F;
  state[static_cast<int>(OutputIndex::BASELINK_POS_I_Y)] = 0.0F;

  // 1. Vehicle yaw exactly M_PI (180 degrees) vs reference yaw 0.0
  state[static_cast<int>(OutputIndex::YAW)] = static_cast<float>(M_PI);
  auto breakdown_pi = evaluateState(state, base_u_, 0);

  EXPECT_FALSE(std::isnan(breakdown_pi.heading));
  EXPECT_FALSE(std::isinf(breakdown_pi.heading));
  EXPECT_FALSE(std::isnan(breakdown_pi.lateral_yaw_error));
  EXPECT_FALSE(std::isinf(breakdown_pi.lateral_yaw_error));
  const float max_heading_cost = params_.heading_coeff * static_cast<float>(M_PI * M_PI);
  EXPECT_NEAR(breakdown_pi.heading, max_heading_cost, 1.0E-3F);

  // 2. Vehicle yaw at -M_PI (-180 degrees)
  state[static_cast<int>(OutputIndex::YAW)] = -static_cast<float>(M_PI);
  auto breakdown_neg_pi = evaluateState(state, base_u_, 0);
  EXPECT_NEAR(breakdown_neg_pi.heading, breakdown_pi.heading, 1.0E-3F);

  // 3. Multi-turn wrapped yaw: 3*PI (equivalent to PI) and -3*PI
  state[static_cast<int>(OutputIndex::YAW)] = 3.0F * static_cast<float>(M_PI);
  auto breakdown_3pi = evaluateState(state, base_u_, 0);
  EXPECT_NEAR(breakdown_3pi.heading, breakdown_pi.heading, 1.0E-3F);

  state[static_cast<int>(OutputIndex::YAW)] = -3.0F * static_cast<float>(M_PI);
  auto breakdown_neg_3pi = evaluateState(state, base_u_, 0);
  EXPECT_NEAR(breakdown_neg_3pi.heading, breakdown_pi.heading, 1.0E-3F);

  // 4. Full circle wrapped yaw: 2*PI (equivalent to 0 error)
  state[static_cast<int>(OutputIndex::YAW)] = 2.0F * static_cast<float>(M_PI);
  auto breakdown_2pi = evaluateState(state, base_u_, 0);
  EXPECT_NEAR(breakdown_2pi.heading, 0.0F, 1.0E-3F);
}

// 12. Singularity Test: Colocated Reference Trajectory Points
// Verifies that consecutive points with identical coordinates do not cause division-by-zero or NaNs.
TEST_F(CostFunctionTest, HandlesColocatedReferencePoints)
{
  std::vector<float> ref_x(NUM_TIMESTEPS_TEST);
  std::vector<float> ref_y(NUM_TIMESTEPS_TEST);
  std::vector<float> ref_yaw(NUM_TIMESTEPS_TEST, 0.0F);
  std::vector<float> ref_v(NUM_TIMESTEPS_TEST, params_.desired_speed);

  for (int i = 0; i < NUM_TIMESTEPS_TEST; ++i) {
    ref_x[i] = static_cast<float>(i);
    ref_y[i] = 0.0F;
  }

  // Create zero-length segments by colocating consecutive points
  ref_x[10] = 10.0F;
  ref_y[10] = 0.0F;
  ref_x[11] = 10.0F;  // Colocated with point 10 (segment length = 0)
  ref_y[11] = 0.0F;
  ref_x[12] = 10.0F;  // Multiple colocated points
  ref_y[12] = 0.0F;

  cost_impl_->setReferenceTrajectory(
    ref_x.data(), ref_y.data(), ref_v.data(), NUM_TIMESTEPS_TEST, ref_yaw.data());

  // Evaluate near and on the colocated points
  output_array state = base_y_;
  state[static_cast<int>(OutputIndex::BASELINK_POS_I_X)] = 10.0F;
  state[static_cast<int>(OutputIndex::BASELINK_POS_I_Y)] = 1.0F;
  state[static_cast<int>(OutputIndex::YAW)] = 0.0F;

  auto breakdown = evaluateState(state, base_u_, 10);

  EXPECT_FALSE(std::isnan(breakdown.lateral_distance));
  EXPECT_FALSE(std::isinf(breakdown.lateral_distance));
  EXPECT_FALSE(std::isnan(breakdown.lateral_yaw_error));
  EXPECT_FALSE(std::isinf(breakdown.lateral_yaw_error));
  EXPECT_FALSE(std::isnan(breakdown.track));
  EXPECT_FALSE(std::isinf(breakdown.track));
  EXPECT_FALSE(std::isnan(breakdown.total));
  EXPECT_FALSE(std::isinf(breakdown.total));

  EXPECT_NEAR(cost_impl_->computeLateralDistanceValue(10.0F, 1.0F), 1.0F, 1.0E-4F);
  EXPECT_FALSE(std::isnan(cost_impl_->computeLateralYawErrorValue(10.0F, 1.0F, 0.2F)));
}

}  // namespace
}  // namespace autoware::mppi_optimizer

