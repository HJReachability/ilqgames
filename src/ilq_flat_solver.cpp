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
#include <ilqgames/solver/lq_solver.h>
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
    std::vector<LinearDynamicsApproximation>* linearization) {
  CHECK_NOTNULL(linearization);

  // Cast dynamics to appropriate type.
  const auto dyn =
      static_cast<const MultiPlayerFlatSystem*>(problem_->Dynamics().get());

  // Populate one timestep at a time.
  for (size_t kk = 0; kk < linearization->size(); kk++)
    (*linearization)[kk] = dyn->LinearizedSystem();
}

float ILQFlatSolver::StateDistance(const VectorXf& x1, const VectorXf& x2,
                                   const std::vector<Dimension>& dims) const {
  const auto& dyn =
      *static_cast<const MultiPlayerFlatSystem*>(problem_->Dynamics().get());

  // If singular return infinite distance and throw a warning. Otherwise, use
  // base class implementation but for nonlinear system states.
  if (dyn.IsLinearSystemStateSingular(x1) ||
      dyn.IsLinearSystemStateSingular(x2)) {
    LOG(WARNING) << "Singular state encountered when computing state distance.";
    return std::numeric_limits<float>::infinity();
  }

  return ILQSolver::StateDistance(dyn.FromLinearSystemState(x1),
                                  dyn.FromLinearSystemState(x2), dims);
}

}  // namespace ilqgames
