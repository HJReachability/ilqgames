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
#include <ilqgames/solver/game_solver.h>
#include <ilqgames/solver/solve_lq_game.h>
#include <ilqgames/utils/linear_dynamics_approximation.h>
#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/quadratic_cost_approximation.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <memory>
#include <vector>

namespace ilqgames {

namespace {

// Multiply all alphas in a set of strategies by the given constant.
void ScaleAlphas(float scaling, std::vector<Strategy>* strategies) {
  CHECK_NOTNULL(strategies);

  for (auto& strategy : *strategies) {
    for (auto& alpha : strategy.alphas) alpha *= scaling;
  }
}

}  // anonymous namespace

void GameSolver::CurrentOperatingPoint(
    const OperatingPoint& last_operating_point,
    const std::vector<Strategy>& current_strategies,
    OperatingPoint* current_operating_point) const {
  CHECK_NOTNULL(current_operating_point);

  // Integrate dynamics and populate operating point, one time step at a time.
  VectorXf x(last_operating_point.xs[0]);
  for (size_t kk = 0; kk < num_time_steps_; kk++) {
    Time t = ComputeTimeStamp(kk);

    // Unpack.
    const VectorXf delta_x = x - last_operating_point.xs[kk];
    const auto& last_us = last_operating_point.us[kk];
    auto& current_us = current_operating_point->us[kk];

    // Record state.
    current_operating_point->xs[kk] = x;

    // Compute and record control for each player.
    for (PlayerIndex jj = 0; jj < dynamics_->NumPlayers(); jj++) {
      const auto& strategy = current_strategies[jj];
      current_us[jj] = strategy(kk, delta_x, last_us[jj]);
    }

    // Integrate dynamics for one time step.
    if (kk < num_time_steps_ - 1)
      x = dynamics_->Integrate(t, time_step_, x, current_us);
  }
}

std::vector<float> GameSolver::EvaluateCosts(const OperatingPoint& op) const {
  std::vector<float> costs(player_costs_.size());
  for (PlayerIndex ii = 0; ii < costs.size(); ii++)
    costs[ii] = player_costs_[ii].Evaluate(op, time_step_);

  return costs;
}

bool GameSolver::HasConverged(
    size_t iteration, const OperatingPoint& last_operating_point,
    const OperatingPoint& current_operating_point) const {
  // Check iterations.
  if (iteration >= params_.max_solver_iters) return true;
  if (iteration == 0) return false;

  // Check operating points.
  return AreOperatingPointsClose(last_operating_point, current_operating_point,
                                 params_.convergence_tolerance);
}

bool GameSolver::AreOperatingPointsClose(const OperatingPoint& op1,
                                         const OperatingPoint& op2,
                                         float threshold) const {
  CHECK_EQ(op1.xs.size(), op2.xs.size());

  for (size_t kk = 0; kk < op1.xs.size(); kk++) {
    if ((op1.xs[kk] - op2.xs[kk]).cwiseAbs().maxCoeff() > threshold)
      return false;
  }

  return true;
}

bool GameSolver::ModifyLQStrategies(
    const OperatingPoint& current_operating_point,
    std::vector<Strategy>* strategies) const {
  // Compute next operating point.
  OperatingPoint next_operating_point(num_time_steps_, dynamics_->NumPlayers(),
                                      current_operating_point.t0);
  CurrentOperatingPoint(current_operating_point, *strategies,
                        &next_operating_point);

  // Initially scale alphas by a fixed amount to avoid unnecessary backtracking.
  ScaleAlphas(params_.initial_alpha_scaling, strategies);

  if (!params_.linesearch) return true;

  // Keep reducing alphas until the maximum elementwise state difference is
  // above a threshold.
  for (size_t ii = 0; ii < params_.max_backtracking_steps; ii++) {
    if (SatisfiesTrustRegion(current_operating_point, next_operating_point))
      return true;

    ScaleAlphas(params_.geometric_alpha_scaling, strategies);
    CurrentOperatingPoint(current_operating_point, *strategies,
                          &next_operating_point);
  }

  LOG(WARNING) << "Exceeded maximum number of backtracking steps.";
  return false;
}

}  // namespace ilqgames
