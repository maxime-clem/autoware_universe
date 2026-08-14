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

#ifndef AUTOWARE__MPPI_OPTIMIZER__DETAIL__TEMPORAL_MPT_NOMINAL_HPP_
#define AUTOWARE__MPPI_OPTIMIZER__DETAIL__TEMPORAL_MPT_NOMINAL_HPP_

#include "autoware/mppi_optimizer/detail/trajectory_utils.hpp"

#include <memory>
#include <optional>
#include <vector>

namespace autoware::mppi_optimizer::detail
{

/**
 * Owns a temporal_mpt::PathTrackingSolver and converts its u-trajectory into MPPI nominal
 * controls (accel_cmd, steer_cmd). Falls back to nullopt when the solve fails.
 */
class TemporalMptNominalSeeder
{
public:
  TemporalMptNominalSeeder();
  ~TemporalMptNominalSeeder();

  TemporalMptNominalSeeder(const TemporalMptNominalSeeder &) = delete;
  TemporalMptNominalSeeder & operator=(const TemporalMptNominalSeeder &) = delete;
  TemporalMptNominalSeeder(TemporalMptNominalSeeder &&) noexcept;
  TemporalMptNominalSeeder & operator=(TemporalMptNominalSeeder &&) noexcept;

  /** Update bicycle lf/lr from ego wheel base (split evenly). */
  void setWheelBase(float wheel_base_m);

  [[nodiscard]] std::optional<std::vector<FirstOrderDubinsMppiControl>> solve(
    const Trajectory & reference, const InitialState & ego,
    const FirstOrderDubinsMppiVehicleParams & vehicle_params, int horizon = kMppiHorizon);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace autoware::mppi_optimizer::detail

#endif  // AUTOWARE__MPPI_OPTIMIZER__DETAIL__TEMPORAL_MPT_NOMINAL_HPP_
