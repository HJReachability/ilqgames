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
// Base class specifying the problem interface for managing calls to the core
// ILQGame solver. Specific examples will be derived from this class.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/solver/game_solver.h>
#include <ilqgames/solver/problem.h>
#include <ilqgames/utils/solver_log.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <algorithm>
#include <memory>
#include <vector>

namespace ilqgames {

std::shared_ptr<SolverLog> Problem::Solve(Time max_runtime) {
  CHECK_NOTNULL(solver_.get());
  CHECK_NOTNULL(strategies_.get());
  CHECK_NOTNULL(operating_point_.get());

  // Create empty log.
  std::shared_ptr<SolverLog> log = CreateNewLog();

  // Solve the problem.
  OperatingPoint final_operating_point(*operating_point_);
  std::vector<Strategy> final_strategies(*strategies_);
  if (!solver_->Solve(x0_, *operating_point_, *strategies_,
                      &final_operating_point, &final_strategies, log.get(),
                      max_runtime)) {
    VLOG(0) << "Solver failed. Not updating operating point and "
               "strategies to failed solution.";
    return log;
  }

  // Store these new strategies/operating point.
  strategies_->swap(final_strategies);
  operating_point_->swap(final_operating_point);

  CHECK_LT((x0_ - operating_point_->xs[0]).cwiseAbs().maxCoeff(),
           constants::kSmallNumber);
  CHECK_LT((x0_ - log->FinalOperatingPoint().xs[0]).cwiseAbs().maxCoeff(),
           constants::kSmallNumber);
  return log;
}

void Problem::SetUpNextRecedingHorizon(const VectorXf& x0, Time t0,
                                       Time planner_runtime) {
  CHECK_NOTNULL(strategies_.get());
  CHECK_NOTNULL(operating_point_.get());
  CHECK_GE(planner_runtime, 0.0);
  CHECK_LE(
      planner_runtime + t0,
      operating_point_->t0 + solver_->TimeStep() * operating_point_->xs.size());
  CHECK_GE(t0, operating_point_->t0);
  CHECK_GE(operating_point_->xs.size(), solver_->NumTimeSteps());

  const MultiPlayerIntegrableSystem& dynamics = solver_->Dynamics();

  // Integrate x0 forward from t0 by approximately planner_runtime to get
  // actual initial state. Integrate up to the next discrete timestep, then
  // integrate for an integer number of discrete timesteps until by the *next*
  // timestep at least 'planner_runtime' has elapsed (done by rounding).
  constexpr float kRoundingError = 0.9;
  const Time relative_t0 = t0 - operating_point_->t0;
  size_t current_timestep =
      static_cast<size_t>(relative_t0 / solver_->TimeStep());
  Time remaining_time_this_step =
      (current_timestep + 1) * solver_->TimeStep() - relative_t0;
  if (remaining_time_this_step < kRoundingError * solver_->TimeStep()) {
    current_timestep += 1;
    remaining_time_this_step = solver_->TimeStep() - remaining_time_this_step;
  }

  CHECK_LT(remaining_time_this_step, solver_->TimeStep());

  // Initially, set x to the integrated version of x0 at the next timestep.
  VectorXf x =
      dynamics.IntegrateToNextTimeStep(t0, x0, *operating_point_, *strategies_);
  operating_point_->t0 = t0 + remaining_time_this_step;
  if (remaining_time_this_step <= planner_runtime) {
    const size_t num_steps_to_integrate = static_cast<size_t>(
        constants::kSmallNumber +  // Add to avoid truncation error.
        (planner_runtime - remaining_time_this_step) / solver_->TimeStep());
    const size_t last_integration_timestep =
        current_timestep + num_steps_to_integrate;

    x = dynamics.Integrate(current_timestep + 1, last_integration_timestep, x,
                           *operating_point_, *strategies_);
    operating_point_->t0 += solver_->TimeStep() * num_steps_to_integrate;
  }

  // Find index of nearest state in the existing plan to this state.
  const auto nearest_iter =
      std::min_element(operating_point_->xs.begin(), operating_point_->xs.end(),
                       [&dynamics, &x](const VectorXf& x1, const VectorXf& x2) {
                         return dynamics.DistanceBetween(x, x1) <
                                dynamics.DistanceBetween(x, x2);
                       });

  // Set initial time to first timestamp in new problem.
  const size_t first_timestep_in_new_problem =
      std::distance(operating_point_->xs.begin(), nearest_iter);

  // Set initial state to this state.
  x0_ = dynamics.Stitch(*nearest_iter, x);
  //  x0_ = *nearest_iter;

  // Update all costs to have the correct initial time.
  Cost::ResetInitialTime(operating_point_->t0);

  // Set final timestep to consider in current operating point.
  const size_t after_final_timestep =
      first_timestep_in_new_problem + solver_->NumTimeSteps();
  const size_t timestep_iterator_end =
      std::min(after_final_timestep, operating_point_->xs.size());

  // Populate strategies and opeating point for the remainder of the
  // existing plan, reusing the old operating point when possible.
  for (size_t kk = first_timestep_in_new_problem; kk < timestep_iterator_end;
       kk++) {
    const size_t kk_new_problem = kk - first_timestep_in_new_problem;

    // Set current state and controls in operating point.
    operating_point_->xs[kk_new_problem].swap(operating_point_->xs[kk]);
    operating_point_->us[kk_new_problem].swap(operating_point_->us[kk]);
    CHECK_EQ(operating_point_->us[kk_new_problem].size(),
             dynamics.NumPlayers());

    // Set current stategy.
    for (auto& strategy : *strategies_) {
      strategy.Ps[kk_new_problem].swap(strategy.Ps[kk]);
      strategy.alphas[kk_new_problem].swap(strategy.alphas[kk]);
    }
  }

  // Make sure operating point is the right size.
  CHECK_GE(operating_point_->xs.size(), solver_->NumTimeSteps());
  if (operating_point_->xs.size() > solver_->NumTimeSteps()) {
    operating_point_->xs.resize(solver_->NumTimeSteps());
    operating_point_->us.resize(solver_->NumTimeSteps());
    for (PlayerIndex ii = 0; ii < dynamics.NumPlayers(); ii++) {
      (*strategies_)[ii].Ps.resize(solver_->NumTimeSteps());
      (*strategies_)[ii].alphas.resize(solver_->NumTimeSteps());
    }
  }

  // Set new operating point controls and strategies to zero and propagate
  // state forward accordingly.
  for (size_t kk = timestep_iterator_end - first_timestep_in_new_problem;
       kk < solver_->NumTimeSteps(); kk++) {
    operating_point_->us[kk].resize(dynamics.NumPlayers());
    for (size_t ii = 0; ii < dynamics.NumPlayers(); ii++) {
      (*strategies_)[ii].Ps[kk].setZero(dynamics.UDim(ii), dynamics.XDim());
      (*strategies_)[ii].alphas[kk].setZero(dynamics.UDim(ii));
      operating_point_->us[kk][ii].setZero(dynamics.UDim(ii));
    }

    operating_point_->xs[kk] = dynamics.Integrate(
        operating_point_->t0 + solver_->ComputeTimeStamp(kk - 1),
        solver_->TimeStep(), operating_point_->xs[kk - 1],
        operating_point_->us[kk - 1]);
  }

  // Invariants.
  CHECK_EQ(operating_point_->xs.size(), solver_->NumTimeSteps());
  CHECK_LE(std::abs(t0 + planner_runtime - operating_point_->t0),
           solver_->TimeStep());
}

void Problem::OverwriteSolution(const OperatingPoint& operating_point,
                                const std::vector<Strategy>& strategies) {
  CHECK_NOTNULL(operating_point_.get());
  CHECK_NOTNULL(strategies_.get());

  *operating_point_ = operating_point;
  *strategies_ = strategies;
}

void Problem::UpdateThresholdTimes(Time time_since_last_update) {
  for (auto& cost : initial_time_costs_)
    cost->IncrementThresholdTime(time_since_last_update);
  for (auto& cost : final_time_costs_)
    cost->IncrementThresholdTime(time_since_last_update);
}

std::shared_ptr<SolverLog> Problem::CreateNewLog() const {
  return std::make_shared<SolverLog>(solver_->TimeStep());
}

}  // namespace ilqgames
