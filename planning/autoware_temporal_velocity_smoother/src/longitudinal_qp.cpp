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

#include "autoware/temporal_velocity_smoother/longitudinal_qp.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace autoware::temporal_velocity_smoother
{

LongitudinalQp::LongitudinalQp(
  const std::size_t horizon, const bool enable_warm_start, const int max_iteration,
  const double eps_abs, const double eps_rel, const bool verbose)
: layout_(horizon),
  solver_(
    std::make_unique<autoware::qp_interface::ProxQPInterface>(
      enable_warm_start, max_iteration, eps_abs, eps_rel, verbose))
{
  if (horizon == 0U) {
    throw std::invalid_argument("LongitudinalQp horizon must be positive");
  }
}

void LongitudinalQp::validate(const QpInput & input) const
{
  const auto n = layout_.n;
  const auto require_size = [n](const auto & values, const char * name) {
    if (values.size() != n) {
      throw std::invalid_argument(std::string{name} + " must contain exactly horizon points");
    }
  };
  require_size(input.reference_s, "reference_s");
  require_size(input.reference_v, "reference_v");
  require_size(input.reference_a, "reference_a");
  require_size(input.velocity_max, "velocity_max");
  require_size(input.tracking_enabled, "tracking_enabled");
  if (!(input.dt > 0.0) || !std::isfinite(input.dt)) {
    throw std::invalid_argument("dt must be finite and positive");
  }
  if (layout_.variables != 8U * n || layout_.constraints != 8U * n) {
    throw std::logic_error("temporal smoother QP dimension invariant was violated");
  }
}

SolveResult LongitudinalQp::solve(const QpInput & input)
{
  validate(input);
  const auto & x = layout_;
  const auto n = x.n;
  Eigen::MatrixXd p = Eigen::MatrixXd::Zero(x.variables, x.variables);
  Eigen::MatrixXd a = Eigen::MatrixXd::Zero(x.constraints, x.variables);
  std::vector<double> q(x.variables, 0.0);
  std::vector<double> lower(x.constraints, -kInfinity);
  std::vector<double> upper(x.constraints, kInfinity);

  for (std::size_t k = 0; k < n; ++k) {
    const double tracking = input.tracking_enabled[k] ? 1.0 : 0.0;
    p(x.v + k, x.v + k) += tracking * input.weights.velocity_tracking;
    q[x.v + k] -= tracking * input.weights.velocity_tracking * input.reference_v[k];
    p(x.a + k, x.a + k) += tracking * input.weights.acceleration_tracking;
    q[x.a + k] -= tracking * input.weights.acceleration_tracking * input.reference_a[k];
    p(x.jerk + k, x.jerk + k) += input.weights.jerk;
    p(x.velocity_slack + k, x.velocity_slack + k) += input.weights.over_velocity;
    p(x.acceleration_slack + k, x.acceleration_slack + k) += input.weights.over_acceleration;
    p(x.jerk_slack + k, x.jerk_slack + k) += input.weights.over_jerk;
    p(x.position_slack + k, x.position_slack + k) += input.weights.over_position;
    q[x.s + k] -= input.weights.progress;
  }

  // A tiny regularizer keeps every neutralized/free direction numerically well posed.
  p.diagonal().array() += 1.0e-9;

  const double dt = input.dt;
  const double dt2 = dt * dt;
  const double dt3 = dt2 * dt;
  for (std::size_t k = 0; k < n; ++k) {
    const auto rs = x.dynamics_row + 3U * k;
    const auto rv = rs + 1U;
    const auto ra = rs + 2U;
    a(rs, x.s + k) = 1.0;
    a(rv, x.v + k) = 1.0;
    a(ra, x.a + k) = 1.0;
    a(rs, x.jerk + k) = -dt3 / 6.0;
    a(rv, x.jerk + k) = -dt2 / 2.0;
    a(ra, x.jerk + k) = -dt;
    if (k == 0U) {
      lower[rs] = upper[rs] = input.initial.s + input.initial.v * dt + 0.5 * input.initial.a * dt2;
      lower[rv] = upper[rv] = input.initial.v + input.initial.a * dt;
      lower[ra] = upper[ra] = input.initial.a;
    } else {
      a(rs, x.s + k - 1U) = -1.0;
      a(rs, x.v + k - 1U) = -dt;
      a(rs, x.a + k - 1U) = -0.5 * dt2;
      a(rv, x.v + k - 1U) = -1.0;
      a(rv, x.a + k - 1U) = -dt;
      a(ra, x.a + k - 1U) = -1.0;
      lower[rs] = upper[rs] = 0.0;
      lower[rv] = upper[rv] = 0.0;
      lower[ra] = upper[ra] = 0.0;
    }

    const auto velocity_row = x.velocity_row + k;
    a(velocity_row, x.v + k) = 1.0;
    a(velocity_row, x.velocity_slack + k) = -1.0;
    lower[velocity_row] = 0.0;
    upper[velocity_row] = std::max(0.0, input.velocity_max[k]);

    const auto acceleration_row = x.acceleration_row + k;
    a(acceleration_row, x.a + k) = 1.0;
    a(acceleration_row, x.acceleration_slack + k) = -1.0;
    lower[acceleration_row] = input.limits.min_acceleration;
    upper[acceleration_row] = input.limits.max_acceleration;

    const auto jerk_row = x.jerk_row + k;
    a(jerk_row, x.jerk + k) = 1.0;
    a(jerk_row, x.jerk_slack + k) = -1.0;
    lower[jerk_row] = input.limits.min_jerk;
    upper[jerk_row] = input.limits.max_jerk;

    const auto position_row = x.position_row + k;
    a(position_row, x.s + k) = 1.0;
    a(position_row, x.position_slack + k) = -1.0;
    if (input.position_limit) {
      upper[position_row] = *input.position_limit;
    }
  }

  // Keep the N-1 row block even when reverse motion is enabled.
  for (std::size_t k = 0; k + 1U < n; ++k) {
    const auto row = x.monotonic_row + k;
    if (!input.allow_reverse) {
      a(row, x.s + k) = 1.0;
      a(row, x.s + k + 1U) = -1.0;
      upper[row] = 0.0;
    }
  }

  a(x.terminal_row, x.v + n - 1U) = 1.0;
  a(x.terminal_row, x.velocity_slack + n - 1U) = -1.0;
  if (input.terminal_velocity_limit) {
    upper[x.terminal_row] = std::max(0.0, *input.terminal_velocity_limit);
  }

  if (q.size() != x.variables || lower.size() != x.constraints || upper.size() != x.constraints) {
    throw std::logic_error("temporal smoother QP vectors violate the 8N invariant");
  }

  SolveResult result;
  const auto solution = solver_->optimize(p, a, q, lower, upper);
  result.success = solver_->isSolved() && solution.size() == x.variables;
  result.status = solver_->getStatus();
  result.iterations = solver_->getIterationNumber();
  if (!result.success) {
    return result;
  }

  result.states.resize(n);
  result.jerk.resize(n);
  result.velocity_slack.resize(n);
  result.acceleration_slack.resize(n);
  result.jerk_slack.resize(n);
  result.position_slack.resize(n);
  for (std::size_t k = 0; k < n; ++k) {
    result.states[k] = {solution[x.s + k], solution[x.v + k], solution[x.a + k]};
    result.jerk[k] = solution[x.jerk + k];
    result.velocity_slack[k] = solution[x.velocity_slack + k];
    result.acceleration_slack[k] = solution[x.acceleration_slack + k];
    result.jerk_slack[k] = solution[x.jerk_slack + k];
    result.position_slack[k] = solution[x.position_slack + k];
  }
  return result;
}

}  // namespace autoware::temporal_velocity_smoother
