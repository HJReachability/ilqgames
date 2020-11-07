/*
 * Copyright (c) 2020, The Regents of the University of California (Regents).
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
// Solver that implements an augmented Lagrangian method. For reference on these
// methods, please refer to Chapter 17 of Nocedal and Wright or the ALTRO paper:
// https://bjack205.github.io/assets/ALTRO.pdf.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/dynamics/multi_player_dynamical_system.h>
#include <ilqgames/dynamics/multi_player_integrable_system.h>
#include <ilqgames/solver/augmented_lagrangian_solver.h>
#include <ilqgames/solver/game_solver.h>
#include <ilqgames/solver/ilq_solver.h>
#include <ilqgames/solver/lq_feedback_solver.h>
#include <ilqgames/solver/lq_open_loop_solver.h>
#include <ilqgames/solver/lq_solver.h>
#include <ilqgames/solver/problem.h>
#include <ilqgames/solver/solver_params.h>
#include <ilqgames/utils/linear_dynamics_approximation.h>
#include <ilqgames/utils/loop_timer.h>
#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/quadratic_cost_approximation.h>
#include <ilqgames/utils/solver_log.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <chrono>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace ilqgames {

std::shared_ptr<SolverLog> AugmentedLagrangianSolver::Solve(bool* success,
                                                            Time max_runtime) {
  if (success) *success = true;

  // Cache initial problem solution so we can restore it at the end.
  const auto& initial_op = problem_->CurrentOperatingPoint();
  const auto& initial_strategies = problem_->CurrentStrategies();

  // Create new log.
  std::shared_ptr<SolverLog> log = CreateNewLog();

  // Determine how much time should be allocated for any individual lower level
  // solver call.
  const Time max_runtime_unconstrained_problem =
      (problem_->IsConstrained())
          ? max_runtime / static_cast<Time>(params_.max_solver_iters)
          : max_runtime;

  // Solve unconstrained problem.
  bool unconstrained_success = false;
  const auto unconstrained_log = unconstrained_solver_->Solve(
      &unconstrained_success, max_runtime_unconstrained_problem);
  log->AddLog(*unconstrained_log);

  VLOG_IF(2, !unconstrained_success)
      << "Unconstrained solver failed on first call.";
  VLOG_IF(2, unconstrained_success)
      << "Unconstrained solver succeeded on first call.";
  if (success) *success &= unconstrained_success;

  // Exit if problem is unconstrained.
  if (!problem_->IsConstrained()) return log;

  // Run until convergence or until the time runs out.
  Time elapsed = max_runtime_unconstrained_problem;
  float max_constraint_error = constants::kInfinity;
  while (log->NumIterates() < params_.max_solver_iters &&
         max_constraint_error > params_.constraint_error_tolerance &&
         elapsed < max_runtime - timer_.RuntimeUpperBound()) {
    // Start loop timer.
    timer_.Tic();

    // Increment multiplers in player costs, and in parallel compute the total
    // squared constraint error.
    max_constraint_error = -constants::kInfinity;
    const OperatingPoint& op = log->FinalOperatingPoint();
    for (auto& pc : problem_->PlayerCosts()) {
      for (size_t kk = 0; kk < op.xs.size(); kk++) {
        const Time t = op.t0 + time::kTimeStep * static_cast<float>(kk);
        const auto& x = op.xs[kk];
        const auto& us = op.us[kk];

        // Scale each lambda.
        for (const auto& constraint : pc.StateConstraints()) {
          const float constraint_error = constraint->Evaluate(t, x);
          max_constraint_error =
              std::max(max_constraint_error, constraint_error);
          constraint->IncrementLambda(t, constraint_error);
        }

        for (const auto& pair : pc.ControlConstraints()) {
          const float constraint_error =
              pair.second->Evaluate(t, us[pair.first]);
          max_constraint_error =
              std::max(max_constraint_error, constraint_error);
          pair.second->IncrementLambda(t, constraint_error);
        }
      }
    }

    // Scale mu.
    Constraint::ScaleMu(params_.geometric_mu_scaling);

    // Log squared constraint violation.
    VLOG(2) << "Max constraint violation at iteration " << log->NumIterates()
            << " is " << max_constraint_error;

    // Update problem solution to make sure we pick up where we left off if the
    // previous unconstrained solver succeeded.
    if (unconstrained_success) {
      problem_->OverwriteSolution(log->FinalOperatingPoint(),
                                  log->FinalStrategies());
    }

    // Run unconstrained solver to convergence. Since we will update problem
    // solutions at each outer iteration, the unconstrained solver should
    // automatically start where it left off.
    const auto unconstrained_log = unconstrained_solver_->Solve(
        &unconstrained_success, max_runtime_unconstrained_problem);

    VLOG_IF(2, unconstrained_success)
        << "Unconstrained solver succeeded on iteration " << log->NumIterates();

    // If we failed then downscale all lambdas and mus for next iteration.
    if (!unconstrained_success) {
      VLOG(2) << "Unconstrained solver failed at iteration "
              << log->NumIterates();
      VLOG(2) << "Downscaling all multipliers.";
      for (auto& pc : problem_->PlayerCosts()) {
        for (const auto& constraint : pc.StateConstraints())
          constraint->ScaleLambdas(params_.geometric_lambda_downscaling);
        for (const auto& pair : pc.ControlConstraints())
          pair.second->ScaleLambdas(params_.geometric_lambda_downscaling);
      }

      Constraint::ScaleMu(params_.geometric_mu_downscaling);
    }

    if (success) *success &= unconstrained_success;
    log->AddLog(*unconstrained_log);

    // Record loop time.
    elapsed += timer_.Toc();
  }

  // If we're still failing constraint satisfaction check mark as failure.
  if (max_constraint_error > params_.constraint_error_tolerance) {
    LOG(WARNING) << "Solver could not satisfy all constraints.";
    if (success) *success = false;
  }

  // Maybe restore initial solution to this problem.
  if (params_.reset_problem)
    problem_->OverwriteSolution(initial_op, initial_strategies);

  // Reset all multipliers.
  if (params_.reset_lambdas) {
    for (auto& pc : problem_->PlayerCosts()) {
      for (const auto& constraint : pc.StateConstraints())
        constraint->ScaleLambdas(constants::kDefaultLambda);
      for (const auto& pair : pc.ControlConstraints())
        pair.second->ScaleLambdas(constants::kDefaultLambda);
    }
  }

  if (params_.reset_mu) Constraint::GlobalMu() = constants::kDefaultMu;

  return log;
}

}  // namespace ilqgames
