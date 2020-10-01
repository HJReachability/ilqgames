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
  // Create new log.
  std::shared_ptr<SolverLog> log = CreateNewLog();

  // Determine how much time should be allocated for any individual lower level
  // solver call.
  const Time max_runtime_unconstrained_problem =
      max_runtime / static_cast<Time>(params_.max_solver_iters);

  // Solve unconstrained problem.
  bool unconstrained_success = false;
  const auto unconstrained_log = unconstrained_solver_->Solve(
      &unconstrained_success, max_runtime_unconstrained_problem);
  log->AddLog(*unconstrained_log);

  LOG_IF(WARNING, !unconstrained_success)
      << "Unconstrained solver failed on first call.";

  // Run until convergence or until the time runs out.
  Time elapsed = 0.0;
  float squared_constraint_error = constants::kInfinity;
  size_t num_iterations = 1;
  while (num_iterations < params_.max_solver_iters &&
         squared_constraint_error <
             params_.squared_constraint_error_tolerance &&
         elapsed < max_runtime - timer_.RuntimeUpperBound()) {
    // Start loop timer.
    timer_.Tic();

    // New iteration.
    VLOG(2) << "Squared constraint violation at iteration " << num_iterations
            << " is " << squared_constraint_error;
    num_iterations++;

    // Predeclare constraint violation.
    float constraint_error;

    // Increment multiplers in player costs, and in parallel compute the total
    // squared constraint error.
    squared_constraint_error = 0.0;
    const OperatingPoint& op = log->FinalOperatingPoint();
    for (auto& pc : problem_->PlayerCosts()) {
      for (size_t kk = 0; kk < op.xs.size(); kk++) {
        const Time t = op.t0 + problem_->TimeStep() * static_cast<float>(kk);
        const auto& x = op.xs[kk];
        const auto& us = op.us[kk];

        // Scale each lambda.
        for (size_t ii = 0; ii < pc.NumStateConstraints(); ii++) {
          const auto& constraint = pc.StateConstraints()[ii];
          constraint->IsSatisfied(t, x, &constraint_error);
          squared_constraint_error += constraint_error * constraint_error;

          auto& lambda = pc.StateLambdas(kk)[ii];
          lambda += pc.Mu() * constraint_error;
        }

        auto constraint_iter = pc.ControlConstraints().begin();
        auto lambda_iter = pc.ControlLambdas(kk).begin();
        while (constraint_iter != pc.ControlConstraints().end()) {
          CHECK(lambda_iter != pc.ControlLambdas(kk).end());
          CHECK_EQ(constraint_iter->first, lambda_iter->first);
          constraint_iter->second->IsSatisfied(t, us[constraint_iter->first],
                                               &constraint_error);
          squared_constraint_error += constraint_error * constraint_error;

          auto& lambda = lambda_iter->second;
          lambda += pc.Mu() * constraint_error;
        }
      }

      // Scale mu.
      pc.ScaleMu(params_.geometric_quadratic_constraint_penalty_scaling);

      // Run unconstrained solver to convergence. Since solvers update problem
      // solutions, the unconstrained solver should automatically start where it
      // left off.
      const auto unconstrained_log = unconstrained_solver_->Solve(
          &unconstrained_success, max_runtime_unconstrained_problem);
      log->AddLog(*unconstrained_log);

      LOG_IF(WARNING, !unconstrained_success)
          << "Unconstrained solver failed at iteration " << num_iterations;
    }
  }

  return log;
}

}  // namespace ilqgames
