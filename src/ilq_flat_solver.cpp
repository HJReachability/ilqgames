/*
 * Copyright (c) 2019, The Regents of the University of California (Regents).
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *    1. Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *    2. Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 *
 *    3. Neither the name of the copyright holder nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Please contact the author(s) of this library if you have any questions.
 * Authors: David Fridovich-Keil   ( dfk@eecs.berkeley.edu )
 */

///////////////////////////////////////////////////////////////////////////////
//
// Base class for all iterative LQ game solvers.
// Structured so that derived classes may only modify the `ModifyLQStrategies`
// and `HasConverged` virtual functions.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/cost/player_cost.h>
#include <ilqgames/solver/ilq_flat_solver.h>
#include <ilqgames/solver/solve_lq_game.h>
#include <ilqgames/utils/linear_dynamics_approximation.h>
#include <ilqgames/utils/loop_timer.h>
#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/quadratic_cost_approximation.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <memory>
#include <vector>

namespace ilqgames {

void ILQFlatSolver::ComputeLinearization(
    const OperatingPoint& op,
    std::vector<LinearDynamicsApproximation>* linearization) {
  CHECK_NOTNULL(linearization);

  // Check if linearization is the right length.
  if (linearization->size() != op.xs.size())
    linearization->resize(op.xs.size());

  // Cast dynamics to appropriate type.
  const auto dyn = static_cast<const MultiPlayerFlatSystem*>(dynamics_.get());

  // Populate one timestep at a time.
  for (size_t kk = 0; kk < op.xs.size(); kk++)
    (*linearization)[kk] = dyn->LinearizedSystem();
}

bool ILQFlatSolver::SatisfiesTrustRegion(
    const OperatingPoint& last_operating_point,
    const OperatingPoint& current_operating_point) const {
  // Check if all states are far from singularity.
  const auto& dyn = *static_cast<const MultiPlayerFlatSystem*>(dynamics_.get());
  for (size_t ii = 0; ii < current_operating_point.xs.size(); ii++) {
    if (dyn.IsLinearSystemStateSingular(current_operating_point.xs[ii])) {
      return false;
    }
  }

  // Are the operating points close.
  return AreOperatingPointsClose(last_operating_point, current_operating_point,
                                 params_.trust_region_size,
                                 params_.trust_region_dimensions);
}

bool ILQFlatSolver::AreOperatingPointsClose(
    const OperatingPoint& op1, const OperatingPoint& op2, float threshold,
    const std::vector<Dimension>& dims) const {
  CHECK_EQ(op1.xs.size(), op2.xs.size());
  const auto& dyn = *static_cast<const MultiPlayerFlatSystem*>(dynamics_.get());

  for (size_t kk = 0; kk < op1.xs.size(); kk++) {
    VectorXf x1 = op1.xs[kk];
    VectorXf x2 = op2.xs[kk];

    // If not singular, use nonlinear system states.
    if (!dyn.IsLinearSystemStateSingular(x1))
      x1 = dyn.FromLinearSystemState(x1);
    if (!dyn.IsLinearSystemStateSingular(x2))
      x2 = dyn.FromLinearSystemState(x2);

    if (dims.empty() && (x1 - x2).cwiseAbs().maxCoeff() > threshold)
      return false;
    else if (!dims.empty()) {
      for (const Dimension dim : dims) {
        if (std::abs(x1(dim) - x2(dim)) > threshold) return false;
      }
    }
  }

  return true;
}

}  // namespace ilqgames
