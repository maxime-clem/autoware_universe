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
  solver_(std::make_unique<autoware::qp_interface::ProxQPInterface>(
    enable_warm_start, max_iteration, eps_abs, eps_rel, verbose)),
  p_(Eigen::MatrixXd::Zero(layout_.variables, layout_.variables)),
  a_(Eigen::MatrixXd::Zero(layout_.constraints, layout_.variables)),
  q_(layout_.variables, 0.0),
  lower_(layout_.constraints, -kInfinity),
  upper_(layout_.constraints, kInfinity)
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
  p_.setZero();
  a_.setZero();
  std::fill(q_.begin(), q_.end(), 0.0);
  std::fill(lower_.begin(), lower_.end(), -kInfinity);
  std::fill(upper_.begin(), upper_.end(), kInfinity);

  for (std::size_t k = 0; k < n; ++k) {
    const double tracking = input.tracking_enabled[k] ? 1.0 : 0.0;
    p_(x.v + k, x.v + k) += tracking * input.weights.velocity_tracking;
    q_[x.v + k] -= tracking * input.weights.velocity_tracking * input.reference_v[k];
    p_(x.a + k, x.a + k) += tracking * input.weights.acceleration_tracking;
    q_[x.a + k] -= tracking * input.weights.acceleration_tracking * input.reference_a[k];
    p_(x.jerk + k, x.jerk + k) += input.weights.jerk;
    p_(x.velocity_slack + k, x.velocity_slack + k) += input.weights.over_velocity;
    p_(x.acceleration_slack + k, x.acceleration_slack + k) += input.weights.over_acceleration;
    p_(x.jerk_slack + k, x.jerk_slack + k) += input.weights.over_jerk;
    p_(x.position_slack + k, x.position_slack + k) += input.weights.over_position;
    q_[x.s + k] -= input.weights.progress;
  }

  // A tiny regularizer keeps every neutralized/free direction numerically well posed.
  p_.diagonal().array() += 1.0e-9;

  const double dt = input.dt;
  const double dt2 = dt * dt;
  const double dt3 = dt2 * dt;
  for (std::size_t k = 0; k < n; ++k) {
    const auto rs = x.dynamics_row + 3U * k;
    const auto rv = rs + 1U;
    const auto ra = rs + 2U;
    a_(rs, x.s + k) = 1.0;
    a_(rv, x.v + k) = 1.0;
    a_(ra, x.a + k) = 1.0;
    a_(rs, x.jerk + k) = -dt3 / 6.0;
    a_(rv, x.jerk + k) = -dt2 / 2.0;
    a_(ra, x.jerk + k) = -dt;
    if (k == 0U) {
      lower_[rs] = upper_[rs] =
        input.initial.s + input.initial.v * dt + 0.5 * input.initial.a * dt2;
      lower_[rv] = upper_[rv] = input.initial.v + input.initial.a * dt;
      lower_[ra] = upper_[ra] = input.initial.a;
    } else {
      a_(rs, x.s + k - 1U) = -1.0;
      a_(rs, x.v + k - 1U) = -dt;
      a_(rs, x.a + k - 1U) = -0.5 * dt2;
      a_(rv, x.v + k - 1U) = -1.0;
      a_(rv, x.a + k - 1U) = -dt;
      a_(ra, x.a + k - 1U) = -1.0;
      lower_[rs] = upper_[rs] = 0.0;
      lower_[rv] = upper_[rv] = 0.0;
      lower_[ra] = upper_[ra] = 0.0;
    }

    const auto velocity_row = x.velocity_row + k;
    a_(velocity_row, x.v + k) = 1.0;
    a_(velocity_row, x.velocity_slack + k) = -1.0;
    lower_[velocity_row] = 0.0;
    upper_[velocity_row] = std::max(0.0, input.velocity_max[k]);

    const auto acceleration_row = x.acceleration_row + k;
    a_(acceleration_row, x.a + k) = 1.0;
    a_(acceleration_row, x.acceleration_slack + k) = -1.0;
    lower_[acceleration_row] = input.limits.min_acceleration;
    upper_[acceleration_row] = input.limits.max_acceleration;

    const auto jerk_row = x.jerk_row + k;
    a_(jerk_row, x.jerk + k) = 1.0;
    a_(jerk_row, x.jerk_slack + k) = -1.0;
    lower_[jerk_row] = input.limits.min_jerk;
    upper_[jerk_row] = input.limits.max_jerk;

    const auto position_row = x.position_row + k;
    if (input.position_limit) {
      a_(position_row, x.s + k) = 1.0;
      a_(position_row, x.position_slack + k) = -1.0;
      upper_[position_row] = *input.position_limit;
    }
  }

  // Keep the N-1 row block even when reverse motion is enabled.
  for (std::size_t k = 0; k + 1U < n; ++k) {
    const auto row = x.monotonic_row + k;
    if (!input.allow_reverse) {
      a_(row, x.s + k) = 1.0;
      a_(row, x.s + k + 1U) = -1.0;
      upper_[row] = 0.0;
    }
  }

  if (input.terminal_velocity_limit) {
    a_(x.terminal_row, x.v + n - 1U) = 1.0;
    a_(x.terminal_row, x.velocity_slack + n - 1U) = -1.0;
    upper_[x.terminal_row] = std::max(0.0, *input.terminal_velocity_limit);
  }

  if (
    q_.size() != x.variables || lower_.size() != x.constraints || upper_.size() != x.constraints) {
    throw std::logic_error("temporal smoother QP vectors violate the 8N invariant");
  }

  SolveResult result;
  const auto solution = solver_->optimize(p_, a_, q_, lower_, upper_);
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
