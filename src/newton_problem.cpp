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

#include <ilqgames/constraint/explicit/dynamic_constraint.h>
#include <ilqgames/constraint/explicit/feedback_constraint.h>
#include <ilqgames/solver/game_solver.h>
#include <ilqgames/solver/newton_problem.h>
#include <ilqgames/solver/problem.h>
#include <ilqgames/utils/relative_time_tracker.h>
#include <ilqgames/utils/solver_log.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

namespace ilqgames {

void NewtonProblem::SetUpNextRecedingHorizon(const VectorXf& x0, Time t0,
                                             Time planner_runtime) {
  CHECK(initialized_);
  auto& op = *operating_point_ref_;

  // Sync to existing problem.
  const size_t first_timestep_in_new_problem =
      SyncToExistingProblem(x0, t0, planner_runtime, op);

  // Set final timestep to consider in current operating point.
  const size_t after_final_timestep =
      first_timestep_in_new_problem + NumTimeSteps();
  const size_t timestep_iterator_end =
      std::min(after_final_timestep, op.xs.size());

  // Populate strategies and opeating point for the remainder of the
  // existing plan, reusing the old operating point when possible.
  for (size_t kk = first_timestep_in_new_problem; kk < timestep_iterator_end;
       kk++) {
    const size_t kk_new_problem = kk - first_timestep_in_new_problem;

    // Set current stategy.
    for (auto& strategy : *strategy_refs_) {
      strategy.Ps[kk_new_problem] = strategy.Ps[kk];
      strategy.alphas[kk_new_problem] = strategy.alphas[kk];
    }

    // Set current state and controls in operating point. To avoid unintended
    // swapping issues with Eigen::Refs just copy.
    op.xs[kk_new_problem] = op.xs[kk];
    for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++) {
      op.us[kk_new_problem][ii] = op.us[kk][ii];

      // Make sure to do this for dual variables too.
      *(*lambda_dyns_)[kk_new_problem][ii] = *(*lambda_dyns_)[kk][ii];
      *(*lambda_feedbacks_)[kk_new_problem][ii] = *(*lambda_feedbacks_)[kk][ii];

      const size_t num_state_constraints =
          player_costs_[ii].NumStateConstraints();
      auto& new_lambdas = (*lambda_state_constraints_)[kk_new_problem][ii];
      auto& old_lambdas = (*lambda_state_constraints_)[kk][ii];
      for (size_t jj = 0; jj < num_state_constraints; jj++) {
        CHECK_EQ(new_lambdas.size(), num_state_constraints);
        CHECK_EQ(old_lambdas.size(), num_state_constraints);
        *new_lambdas[jj] = *old_lambdas[jj];
      }

      CHECK_EQ((*lambda_control_constraints_)[kk_new_problem][ii].size(),
               (*lambda_control_constraints_)[kk][ii].size());
      auto new_iter =
          (*lambda_control_constraints_)[kk_new_problem][ii].begin();
      auto old_iter = (*lambda_control_constraints_)[kk][ii].begin();
      for (; new_iter !=
                 (*lambda_control_constraints_)[kk_new_problem][ii].end() &&
             old_iter != (*lambda_control_constraints_)[kk][ii].end();
           new_iter++, old_iter++) {
        CHECK_EQ(new_iter->first, old_iter->first);
        *new_iter->second = *old_iter->second;
      }
    }
  }

  // Make sure operating point is the right size.
  CHECK_EQ(op.xs.size(), NumTimeSteps());

  // Set new operating point controls and strategies to zero and propagate
  // state forward accordingly. Set new dual variables to zero.
  for (size_t kk = timestep_iterator_end - first_timestep_in_new_problem;
       kk < NumTimeSteps(); kk++) {
    // First integrate the previous state and controls.
    op.xs[kk] =
        dynamics_->Integrate(InitialTime() + ComputeRelativeTimeStamp(kk - 1),
                             time_step_, op.xs[kk - 1], op.us[kk - 1]);

    // Now update everything per player.
    for (size_t ii = 0; ii < dynamics_->NumPlayers(); ii++) {
      auto& strategy = (*strategy_refs_)[ii];
      strategy.Ps[kk].setZero();
      strategy.alphas[kk].setZero();
      op.us[kk][ii].setZero();

      // Make sure to do this for dual variables too.
      *(*lambda_dyns_)[kk][ii] = 0.0;
      *(*lambda_feedbacks_)[kk][ii] = 0.0;

      for (auto& lambda_state : (*lambda_state_constraints_)[kk][ii])
        *lambda_state = 0.0;

      for (auto& pair : (*lambda_control_constraints_)[kk][ii])
        *pair.second = 0.0;
    }
  }
}

size_t NewtonProblem::NumPrimals() const {
  CHECK(initialized_);

  // Start by computing the number of shared variables - states and controls.
  size_t total = NumOperatingPointVariables();

  // Accumulate players' individual strategy parameters.
  for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++)
    total += (*strategy_refs_)[ii].NumVariables();

  return total;
}

size_t NewtonProblem::NumOperatingPointVariables() const {
  CHECK(initialized_);

  return NumTimeSteps() * (dynamics_->XDim() + dynamics_->TotalUDim());
}

size_t NewtonProblem::NumDuals() const {
  CHECK(initialized_);

  // Start by computing the number of initial state multipliers and dynamics
  // multipliers and feedback multipliers (which should be the same).
  size_t total = dynamics_->NumPlayers() * (2 * num_time_steps_ + 1);

  for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++) {
    const auto& player_cost = player_costs_[ii];

    // Accumulate the constraint multipliers for each state constraint.
    total += player_cost.NumStateConstraints() * num_time_steps_;

    // Accumulate the control constraint multipliers
    total += player_cost.NumControlConstraints() * num_time_steps_;
  }

  return total;
}

size_t NewtonProblem::KKTSystemSize() const {
  CHECK(initialized_);

  // Count the number of equations in the overall KKT system comprising all
  // players' problems. To start, the gradient of the Lagrangian has as many
  // dimensions as there are primal variables.
  size_t total = NumPrimals();

  // Each constraint (for each player) has a dual variable associated with it,
  // so the contribution from constraints is just the number of dual
  // variables.
  total += NumDuals();

  return total;
}

void NewtonProblem::ConstructPrimalsAndDuals() {
  // Handle primals.
  primals_.setZero(NumPrimals());
  ConstructInitialOperatingPoint();
  ConstructInitialStrategies();

  // Handle duals.
  duals_.setZero(NumDuals());
  ConstructInitialLambdas();
}

void NewtonProblem::ConstructInitialOperatingPoint() {
  operating_point_ref_.reset(
      new OperatingPointRef(num_time_steps_, 0.0, dynamics_, primals_));
  operating_point_.reset(new OperatingPoint(*operating_point_ref_));
}

void NewtonProblem::ConstructInitialStrategies() {
  strategy_refs_.reset(new std::vector<StrategyRef>());
  size_t primal_idx = NumOperatingPointVariables();
  for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++) {
    strategy_refs_->emplace_back(num_time_steps_, dynamics_->XDim(),
                                 dynamics_->UDim(ii), primals_, primal_idx);
    primal_idx += strategies_->back().NumVariables();
  }
}

void NewtonProblem::ConstructInitialLambdas() {
  lambda_dyns_.reset(new std::vector<std::vector<float*>>(num_time_steps_));
  lambda_feedbacks_.reset(
      new std::vector<std::vector<float*>>(num_time_steps_));
  lambda_state_constraints_.reset(
      new std::vector<std::vector<std::vector<float*>>>(num_time_steps_));
  lambda_control_constraints_.reset(
      new std::vector<std::vector<PlayerDualMap>>(num_time_steps_));

  // Keep the ordering broken out by timesteps. Outer index is always time,
  // and inner one is player ID.
  size_t dual_idx = 0;
  for (size_t kk = 0; kk < num_time_steps_; kk++) {
    // Preallocate memory for state and control constraints for each player.
    (*lambda_state_constraints_)[kk].resize(dynamics_->NumPlayers());
    (*lambda_control_constraints_)[kk].resize(dynamics_->NumPlayers());

    // Populate for each player.
    for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++) {
      (*lambda_dyns_)[kk].emplace_back(&duals_(dual_idx));
      dual_idx++;

      (*lambda_feedbacks_)[kk].emplace_back(&duals_(dual_idx));
      dual_idx++;

      // Add a separate dual variable for each state constraint for this
      // player at this time.
      for (const auto& c : player_costs_[ii].StateConstraints()) {
        (*lambda_state_constraints_)[kk][ii].emplace_back(&duals_(dual_idx));
        dual_idx++;
      }

      // Do likewise for control costs, though they are stored in a different
      // data structure.
      for (const auto& pair : player_costs_[ii].ControlConstraints()) {
        (*lambda_control_constraints_)[kk][ii].emplace(pair.first,
                                                       &duals_(dual_idx));
        dual_idx++;
      }
    }
  }
}

void NewtonProblem::ConstructDynamicConstraint() {
  dynamic_constraint_.reset(
      new DynamicConstraint(&NormalDynamics(), "Dynamic constraint"));
}

void NewtonProblem::ConstructFeedbackConstraints() {
  feedback_constraints_.reset(new std::vector<FeedbackConstraint>());
  for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++)
    feedback_constraints_->emplace_back(
        &(*strategy_refs_)[ii],
        "Feedback Constraint (" + std::to_string(ii) + ")");
}

}  // namespace ilqgames
