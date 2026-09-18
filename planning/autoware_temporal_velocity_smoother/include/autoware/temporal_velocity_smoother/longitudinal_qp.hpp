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

#ifndef AUTOWARE__TEMPORAL_VELOCITY_SMOOTHER__LONGITUDINAL_QP_HPP_
#define AUTOWARE__TEMPORAL_VELOCITY_SMOOTHER__LONGITUDINAL_QP_HPP_

#include "autoware/temporal_velocity_smoother/types.hpp"

#include <Eigen/Core>
#include <autoware/qp_interface/proxqp_interface.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace autoware::temporal_velocity_smoother
{

class LongitudinalQp
{
public:
  LongitudinalQp(
    std::size_t horizon, bool enable_warm_start, int max_iteration, double eps_abs, double eps_rel,
    bool verbose = false);

  [[nodiscard]] const ProblemLayout & layout() const { return layout_; }
  SolveResult solve(const QpInput & input);

private:
  void validate(const QpInput & input) const;

  ProblemLayout layout_;
  std::unique_ptr<autoware::qp_interface::ProxQPInterface> solver_;
  Eigen::MatrixXd p_;
  Eigen::MatrixXd a_;
  std::vector<double> q_;
  std::vector<double> lower_;
  std::vector<double> upper_;
};

}  // namespace autoware::temporal_velocity_smoother

#endif  // AUTOWARE__TEMPORAL_VELOCITY_SMOOTHER__LONGITUDINAL_QP_HPP_
