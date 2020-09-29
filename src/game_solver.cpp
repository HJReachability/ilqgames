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
#include <ilqgames/dynamics/multi_player_dynamical_system.h>
#include <ilqgames/solver/game_solver.h>
#include <ilqgames/solver/lq_solver.h>
#include <ilqgames/utils/linear_dynamics_approximation.h>
#include <ilqgames/utils/loop_timer.h>
#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/quadratic_cost_approximation.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <algorithm>
#include <memory>
#include <vector>

namespace ilqgames {

void GameSolver::ComputeLinearization(
    const OperatingPoint& op,
    std::vector<LinearDynamicsApproximation>* linearization) {
  CHECK_NOTNULL(linearization);

  // Check if linearization is the right length.
  if (linearization->size() != op.xs.size())
    linearization->resize(op.xs.size());

  // Cast dynamics to appropriate type.
  const auto dyn = static_cast<const MultiPlayerDynamicalSystem*>(
      problem_->Dynamics().get());

  // Populate one timestep at a time.
  for (size_t kk = 0; kk < op.xs.size(); kk++) {
    const Time t =
        problem_->InitialTime() + problem_->ComputeRelativeTimeStamp(kk);
    (*linearization)[kk] = dyn->Linearize(t, op.xs[kk], op.us[kk]);
  }
}

void GameSolver::ComputeLinearization(
    std::vector<LinearDynamicsApproximation>* linearization) {
  CHECK_NOTNULL(linearization);

  // Cast dynamics to appropriate type and make sure the system is linearizable.
  CHECK(problem_->Dynamics()->TreatAsLinear());
  const auto& dyn = problem_->FlatDynamics();

  // Populate one timestep at a time.
  for (size_t kk = 0; kk < linearization->size(); kk++)
    (*linearization)[kk] = dyn.LinearizedSystem();
}

void GameSolver::ComputeCostQuadraticization(
    const OperatingPoint& op,
    std::vector<std::vector<QuadraticCostApproximation>>* q) {
  for (size_t kk = 0; kk < problem_->NumTimeSteps(); kk++) {
    const Time t =
        problem_->InitialTime() + problem_->ComputeRelativeTimeStamp(kk);
    const auto& x = op.xs[kk];
    const auto& us = op.us[kk];

    // Quadraticize costs.
    for (PlayerIndex ii = 0; ii < problem_->Dynamics()->NumPlayers(); ii++) {
      const PlayerCost& cost = problem_->PlayerCosts()[ii];

      if (cost.IsTimeAdditive() ||
          problem_->PlayerCosts()[ii].TimeOfExtremeCost() == kk)
        (*q)[kk][ii] = cost.Quadraticize(t, x, us);
      else
        (*q)[kk][ii] = cost.QuadraticizeBarriersAndControlCosts(t, x, us);
    }
  }
}

}  // namespace ilqgames
