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

#include <ilqgames/solver/ilq_solver.h>
#include <ilqgames/solver/problem.h>
#include <ilqgames/utils/solver_log.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <memory>
#include <vector>

namespace ilqgames {

std::shared_ptr<SolverLog> Problem::Solve() {
  CHECK_NOTNULL(solver_.get());
  CHECK_NOTNULL(strategies_.get());
  CHECK_NOTNULL(operating_point_.get());

  // Create empty log.
  std::shared_ptr<SolverLog> log = CreateNewLog();

  // Solver the problem.
  std::vector<Strategy> final_strategies(*strategies_);
  OperatingPoint final_operating_point(*operating_point_);
  if (!solver_->Solve(x0_, *operating_point_, *strategies_,
                      &final_operating_point, &final_strategies, log.get())) {
    LOG(WARNING) << "Solver failed.";
    return nullptr;
  }

  // Store these new strategies/operating point.
  strategies_->swap(final_strategies);
  operating_point_->swap(final_operating_point);

  LOG(INFO) << "Solver succeeded.";
  return log;
}

void Problem::ResetInitialConditions(const VectorXf& x0, Time t0,
                                     Time planner_runtime) {
  CHECK_NOTNULL(strategies_.get());
  CHECK_NOTNULL(operating_point_.get());
  CHECK_GT(planner_runtime, 0.0);
  CHECK_LT(planner_runtime + t0, solver_->TimeHorizon());
  CHECK_GE(t0, 0.0);

  const MultiPlayerDynamicalSystem& dynamics = solver_->Dynamics();

  // Integrate x0 forward from t0 by approximately planner_runtime to get
  // actual initial state. Integrate up to the next discrete timestep, then
  // integrate for an integer number of discrete timesteps until at least
  // 'planner_runtime' has elapsed.
  const Time relative_t0 = t0 - operating_point_->t0;
  const size_t current_timestep =
      static_cast<size_t>(relative_t0 / solver_->TimeStep());
  const Time remaining_time_this_step =
      solver_->TimeStep() * (current_timestep + 1) - relative_t0;
  const size_t num_steps_to_integrate =
      1 + static_cast<size_t>((planner_runtime - remaining_time_this_step) /
                              solver_->TimeStep());
  const size_t first_timestep_in_new_problem =
      current_timestep + 1 + num_steps_to_integrate;

  VectorXf x = dynamics.IntegrateToNextTimeStep(
      t0, solver_->TimeStep(), x0, *operating_point_, *strategies_);
  x = dynamics.Integrate(current_timestep + 1, first_timestep_in_new_problem,
                         solver_->TimeStep(), x, *operating_point_,
                         *strategies_);

  // Set initial state to this state.
  x0_ = x;

  // Set initial time to first timestamp in new problem.
  operating_point_->t0 +=
      solver_->ComputeTimeStamp(first_timestep_in_new_problem);

  // Populate strategies and opeating point for the remainder of the
  // existing plan, reusing the old operating point when possible.
  for (size_t kk = first_timestep_in_new_problem;
       kk < operating_point_->xs.size(); kk++) {
    const size_t kk_new_problem = kk - first_timestep_in_new_problem;

    // Set current state and controls in operating point.
    operating_point_->xs[kk_new_problem] = operating_point_->xs[kk];
    operating_point_->us[kk_new_problem].swap(operating_point_->us[kk]);

    // Set current stategy.
    for (auto& strategy : *strategies_) {
      strategy.Ps[kk_new_problem] = strategy.Ps[kk];
      strategy.alphas[kk_new_problem] = strategy.alphas[kk];
    }
  }

  // Set new operating point controls and strategies to zero and propagate state
  // forward accordingly.
  for (size_t kk = operating_point_->xs.size() - first_timestep_in_new_problem;
       kk < operating_point_->xs.size(); kk++) {
    for (size_t ii = 0; ii < dynamics.NumPlayers(); ii++) {
      (*strategies_)[ii].Ps[kk].setZero();
      (*strategies_)[ii].alphas[kk].setZero();
      operating_point_->us[kk][ii].setZero();
    }

    operating_point_->xs[kk] = dynamics.Integrate(
        operating_point_->t0 + solver_->ComputeTimeStamp(kk - 1),
        solver_->TimeStep(), operating_point_->xs[kk - 1],
        operating_point_->us[kk - 1]);
  }
}

std::shared_ptr<SolverLog> Problem::CreateNewLog() const {
  return std::make_shared<SolverLog>(solver_->TimeStep());
}

}  // namespace ilqgames
