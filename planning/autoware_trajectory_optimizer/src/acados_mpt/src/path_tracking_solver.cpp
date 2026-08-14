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

#include "temporal_mpt/path_tracking_solver.hpp"

#include "acados_interface.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace temporal_mpt
{
namespace
{
constexpr double kTwoPi = 2.0 * M_PI;
}  // namespace

struct PathTrackingSolver::Impl
{
  AcadosInterface solver;
  double lf{1.0};
  double lr{1.0};
};

PathTrackingSolver::PathTrackingSolver() : impl_(std::make_unique<Impl>())
{
}

PathTrackingSolver::~PathTrackingSolver() = default;

PathTrackingSolver::PathTrackingSolver(PathTrackingSolver &&) noexcept = default;

PathTrackingSolver & PathTrackingSolver::operator=(PathTrackingSolver &&) noexcept = default;

void PathTrackingSolver::setModelParameters(const double lf, const double lr)
{
  impl_->lf = lf;
  impl_->lr = lr;
}

std::size_t PathTrackingSolver::horizon()
{
  return N;
}

PathTrackingResult PathTrackingSolver::solve(
  const PathTrackingInitialState & x0, const PathTrackingReference & reference)
{
  PathTrackingResult result;
  result.accel_cmd.assign(N, 0.0);
  result.steer_cmd.assign(N, 0.0);

  const std::size_t n_pts =
    std::min({reference.x.size(), reference.y.size(), reference.yaw.size(), reference.v.size()});
  if (n_pts < 2U) {
    result.status = -2;
    return result;
  }

  const std::array<double, NP> model_params = {impl_->lf, impl_->lr};
  impl_->solver.setParametersAllStages(model_params);

  const std::array<double, NX> x0_world = {x0.x, x0.y, x0.yaw, std::max(0.0, x0.v)};

  size_t start_idx = 0;
  {
    double best_d2 = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < n_pts; ++i) {
      const double dx = reference.x[i] - x0_world[0];
      const double dy = reference.y[i] - x0_world[1];
      const double d2 = dx * dx + dy * dy;
      if (d2 < best_d2) {
        best_d2 = d2;
        start_idx = i;
      }
    }
  }

  const double yaw_at_start = reference.yaw[start_idx];
  const double psi_bias = std::round((x0_world[2] - yaw_at_start) / kTwoPi) * kTwoPi;
  const double x_off = x0_world[0];
  const double y_off = x0_world[1];

  for (size_t k = 0; k < N; ++k) {
    if (k == 0) {
      const std::array<double, NY> yref = {
        x0_world[0] - x_off, x0_world[1] - y_off, x0_world[2], x0_world[3], 0.0, 0.0};
      impl_->solver.setStageReference(static_cast<int>(k), yref);
      continue;
    }
    const size_t idx = std::min(start_idx + k, n_pts - 1);
    const double yaw = reference.yaw[idx] + psi_bias;
    const double v_ref = std::max(0.0, reference.v[idx]);
    const std::array<double, NY> yref = {
      reference.x[idx] - x_off, reference.y[idx] - y_off, yaw, v_ref, 0.0, 0.0};
    impl_->solver.setStageReference(static_cast<int>(k), yref);
  }

  const size_t terminal_idx = std::min(start_idx + N, n_pts - 1);
  const double terminal_yaw = reference.yaw[terminal_idx] + psi_bias;
  const double terminal_v_ref = std::max(0.0, reference.v[terminal_idx]);
  impl_->solver.setTerminalReference(
    {reference.x[terminal_idx] - x_off, reference.y[terminal_idx] - y_off, terminal_yaw,
     terminal_v_ref});

  const std::array<double, NX> x0_local = {
    x0_world[0] - x_off, x0_world[1] - y_off, x0_world[2], x0_world[3]};
  const AcadosSolution solution = impl_->solver.getControl(x0_local);
  result.status = solution.status;
  result.ok = (solution.status == 0);
  if (!result.ok) {
    return result;
  }

  for (size_t k = 0; k < N; ++k) {
    result.accel_cmd[k] = solution.utraj[k][0];
    result.steer_cmd[k] = solution.utraj[k][1];
  }
  return result;
}

}  // namespace temporal_mpt
