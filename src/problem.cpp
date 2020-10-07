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
#include <ilqgames/utils/relative_time_tracker.h>
#include <ilqgames/utils/solver_log.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <algorithm>
#include <memory>
#include <vector>

// // Time horizon and step.
// DEFINE_double(time_horizon, 10.0, "Total time horizon (s).");
// DEFINE_double(time_step, 0.1, "Length of discrete time step (s).");

namespace ilqgames {

Problem::Problem() : initialized_(false) {}

size_t Problem::SyncToExistingProblem(const VectorXf& x0, Time t0,
                                      Time planner_runtime,
                                      OperatingPoint& op) {
  CHECK(initialized_);
  CHECK_GE(planner_runtime, 0.0);
  CHECK_LE(planner_runtime + t0, operating_point_->t0 + time::kTimeHorizon);
  CHECK_GE(t0, operating_point_->t0);

  // Integrate x0 forward from t0 by approximately planner_runtime to get
  // actual initial state. Integrate up to the next discrete timestep, then
  // integrate for an integer number of discrete timesteps until by the *next*
  // timestep at least 'planner_runtime' has elapsed (done by rounding).
  constexpr float kRoundingError = 0.9;
  const Time relative_t0 = t0 - op.t0;
  size_t current_timestep = static_cast<size_t>(relative_t0 / time::kTimeStep);
  Time remaining_time_this_step =
      (current_timestep + 1) * time::kTimeStep - relative_t0;
  if (remaining_time_this_step < kRoundingError * time::kTimeStep) {
    current_timestep += 1;
    remaining_time_this_step = time::kTimeStep - remaining_time_this_step;
  }

  CHECK_LT(remaining_time_this_step, time::kTimeStep);

  // Initially, set x to the integrated version of x0 at the next timestep.
  VectorXf x = dynamics_->IntegrateToNextTimeStep(t0, x0, *operating_point_,
                                                  *strategies_);
  op.t0 = t0 + remaining_time_this_step;
  if (remaining_time_this_step <= planner_runtime) {
    const size_t num_steps_to_integrate = static_cast<size_t>(
        constants::kSmallNumber +  // Add to avoid truncation error.
        (planner_runtime - remaining_time_this_step) / time::kTimeStep);
    const size_t last_integration_timestep =
        current_timestep + num_steps_to_integrate;

    x = dynamics_->Integrate(current_timestep + 1, last_integration_timestep, x,
                             *operating_point_, *strategies_);
    op.t0 += time::kTimeStep * num_steps_to_integrate;
  }

  // Find index of nearest state in the existing plan to this state.
  const auto nearest_iter =
      std::min_element(op.xs.begin(), op.xs.end(),
                       [this, &x](const VectorXf& x1, const VectorXf& x2) {
                         return dynamics_->DistanceBetween(x, x1) <
                                dynamics_->DistanceBetween(x, x2);
                       });

  // Set initial time to first timestamp in new problem.
  const size_t first_timestep_in_new_problem =
      std::distance(op.xs.begin(), nearest_iter);

  // Set initial state to this state.
  x0_ = dynamics_->Stitch(*nearest_iter, x);

  // Update all costs to have the correct initial time.
  RelativeTimeTracker::ResetInitialTime(op.t0);

  // Check an invariant.
  CHECK_LE(std::abs(t0 + planner_runtime - op.t0), time::kTimeStep);
  return first_timestep_in_new_problem;
}

void Problem::SetUpNextRecedingHorizon(const VectorXf& x0, Time t0,
                                       Time planner_runtime) {
  CHECK(initialized_);

  // Sync to existing problem.
  const size_t first_timestep_in_new_problem =
      SyncToExistingProblem(x0, t0, planner_runtime, *operating_point_);

  // Set final timestep to consider in current operating point.
  const size_t after_final_timestep =
      first_timestep_in_new_problem + time::kNumTimeSteps;
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
             dynamics_->NumPlayers());

    // Set current stategy.
    for (auto& strategy : *strategies_) {
      strategy.Ps[kk_new_problem].swap(strategy.Ps[kk]);
      strategy.alphas[kk_new_problem].swap(strategy.alphas[kk]);
    }
  }

  // Make sure operating point is the right size.
  CHECK_GE(operating_point_->xs.size(), time::kNumTimeSteps);
  if (operating_point_->xs.size() > time::kNumTimeSteps) {
    operating_point_->xs.resize(time::kNumTimeSteps);
    operating_point_->us.resize(time::kNumTimeSteps);
    for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++) {
      (*strategies_)[ii].Ps.resize(time::kNumTimeSteps);
      (*strategies_)[ii].alphas.resize(time::kNumTimeSteps);
    }
  }

  // Set new operating point controls and strategies to zero and propagate
  // state forward accordingly.
  for (size_t kk = timestep_iterator_end - first_timestep_in_new_problem;
       kk < time::kNumTimeSteps; kk++) {
    operating_point_->us[kk].resize(dynamics_->NumPlayers());
    for (size_t ii = 0; ii < dynamics_->NumPlayers(); ii++) {
      (*strategies_)[ii].Ps[kk].setZero(dynamics_->UDim(ii), dynamics_->XDim());
      (*strategies_)[ii].alphas[kk].setZero(dynamics_->UDim(ii));
      operating_point_->us[kk][ii].setZero(dynamics_->UDim(ii));
    }

    operating_point_->xs[kk] = dynamics_->Integrate(
        RelativeTimeTracker::RelativeTime(kk - 1), time::kTimeStep,
        operating_point_->xs[kk - 1], operating_point_->us[kk - 1]);
  }
}

void Problem::OverwriteSolution(const OperatingPoint& operating_point,
                                const std::vector<Strategy>& strategies) {
  CHECK(initialized_);

  *operating_point_ = operating_point;
  *strategies_ = strategies;
}

bool Problem::IsConstrained() const {
  for (const auto& pc : player_costs_) {
    if (pc.IsConstrained()) return true;
  }

  return false;
}

}  // namespace ilqgames
