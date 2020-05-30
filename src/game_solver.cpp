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

#include <ilqgames/constraint/constraint.h>
#include <ilqgames/cost/player_cost.h>
#include <ilqgames/solver/game_solver.h>
#include <ilqgames/solver/lq_solver.h>
#include <ilqgames/utils/compute_strategy_costs.h>
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

namespace {

// Multiply all alphas in a set of strategies by the given constant.
void ScaleAlphas(float scaling, std::vector<Strategy>* strategies) {
  CHECK_NOTNULL(strategies);

  for (auto& strategy : *strategies) {
    for (auto& alpha : strategy.alphas) alpha *= scaling;
  }
}

}  // anonymous namespace

bool GameSolver::Solve(const VectorXf& x0,
                       const OperatingPoint& initial_operating_point,
                       const std::vector<Strategy>& initial_strategies,
                       OperatingPoint* final_operating_point,
                       std::vector<Strategy>* final_strategies, SolverLog* log,
                       Time max_runtime) {
  // Start a stopwatch.
  const auto solver_call_time = clock::now();

  // Keep iterating until convergence.
  auto elapsed_time = [](const std::chrono::time_point<clock>& start) {
    return std::chrono::duration<Time>(clock::now() - start).count();
  };  // elapsed_time

  // Chech return pointers not null.
  CHECK_NOTNULL(final_strategies);
  CHECK_NOTNULL(final_operating_point);

  // Make sure we have enough strategies for each time step.
  DCHECK_EQ(dynamics_->NumPlayers(), initial_strategies.size());
  DCHECK(std::accumulate(
      initial_strategies.begin(), initial_strategies.end(), true,
      [this](bool correct_so_far, const Strategy& strategy) {
        return correct_so_far &=
               strategy.Ps.size() == this->num_time_steps_ &&
               strategy.alphas.size() == this->num_time_steps_;
      }));

  // Flag for whether or not the initial operating point is all zeros.
  // If it is all zeros, we will populate it with the initial state unrolled
  // with the initial strategies in the first iteration of the solver, but
  // otherwise we will leave it alone.
  const bool is_initial_operating_point_zero =
      initial_operating_point.xs[0].squaredNorm() < constants::kSmallNumber;

  // Last and current operating points. Make sure the last one starts from the
  // current state so that the current one will start there as well.
  // NOTE: setting the current operating point to start at x0 is critical to the
  // constraint satisfaction check at the first iteration.
  OperatingPoint last_operating_point(initial_operating_point);
  OperatingPoint current_operating_point(initial_operating_point);
  current_operating_point.xs[0] = x0;
  last_operating_point.xs[0] = x0;

  // Current strategies.
  std::vector<Strategy> current_strategies(initial_strategies);

  // Reset all constraint barrier weights to unity.
  for (PlayerCost& cost : player_costs_) cost.ResetConstraintBarrierWeights();

  // Things to keep track of during each iteration.
  size_t num_iterations = 0;
  size_t num_iterations_since_barrier_rescaling = 0;
  bool has_converged = false;

  // Turn constraints on.
  auto turn_constraints_on = [this]() {
    for (auto& cost : player_costs_) cost.TurnConstraintsOn();
  };  // turn_constraints_on

  auto turn_constraints_off = [this]() {
    for (auto& cost : player_costs_) cost.TurnConstraintsOff();
  };  // turn_constraints_on

  // Swap operating points and compute new current operating point. Future
  // operating points will be computed during the call to `ModifyLQStrategies`
  // which occurs after solving the LQ game.
  bool was_operating_point_feasible;
  std::vector<float> total_costs;

  last_operating_point.swap(current_operating_point);
  CurrentOperatingPoint(last_operating_point, current_strategies,
                        &current_operating_point, &has_converged, &total_costs,
                        false, &was_operating_point_feasible);

  // Log current iterate.
  if (log) {
    log->AddSolverIterate(current_operating_point, current_strategies,
                          total_costs, elapsed_time(solver_call_time),
                          has_converged);
  }

  // Main loop with timer for anytime execution.
  while (num_iterations < params_.max_solver_iters && !has_converged &&
         elapsed_time(solver_call_time) <
             max_runtime - timer_.RuntimeUpperBound()) {
    // Start loop timer.
    timer_.Tic();

    // New iteration.
    num_iterations++;
    num_iterations_since_barrier_rescaling++;

    // Maybe rescale constraint barrier weights.
    if (num_iterations_since_barrier_rescaling >
        params_.barrier_scaling_iters) {
      num_iterations_since_barrier_rescaling = 0;
      for (PlayerCost& cost : player_costs_)
        cost.ScaleConstraintBarrierWeights(params_.geometric_barrier_scaling);
    }

    // If operating point is feasible, turn on constraints. If it is
    // not feasible, then turn them off.
    if (was_operating_point_feasible)
      turn_constraints_on();
    else
      turn_constraints_off();

    // Linearize dynamics and quadraticize costs for all players about the new
    // operating point, only if the system can't be treated as linear from the
    // outset, in which case we've already linearized it.
    if (!dynamics_->TreatAsLinear())
      ComputeLinearization(current_operating_point, &linearization_);

    for (size_t kk = 0; kk < num_time_steps_; kk++) {
      const Time t = initial_operating_point.t0 + ComputeTimeStamp(kk);
      const auto& x = current_operating_point.xs[kk];
      const auto& us = current_operating_point.us[kk];

      // Quadraticize costs.
      std::transform(player_costs_.begin(), player_costs_.end(),
                     quadraticization_[kk].begin(),
                     [&t, &x, &us](const PlayerCost& cost) {
                       return cost.Quadraticize(t, x, us);
                     });
    }

    // Solve LQ game.
    current_strategies =
        lq_solver_->Solve(linearization_, quadraticization_, x0);

    // Modify this LQ solution.
    if (!ModifyLQStrategies(&current_strategies, &current_operating_point,
                            &was_operating_point_feasible, &has_converged,
                            &total_costs)) {
      // Maybe emit warning if exiting early.
      if (num_iterations == 1) {
        VLOG(1)
            << "Solver exited after during first iteration, which may indicate "
               "an infeasible initial operating point.";

        if (was_operating_point_feasible)
          VLOG(1) << "Previous operating point was feasible.";
        else {
          VLOG(1) << "Previous operating point was infeasible.";
        }
      }

      return false;
    }

    // Log current iterate.
    if (log) {
      log->AddSolverIterate(current_operating_point, current_strategies,
                            total_costs, elapsed_time(solver_call_time),
                            has_converged);
    }

    // Record loop runtime.
    timer_.Toc();
  }

  // Maybe emit warning if exiting early.
  if (num_iterations == 1) {
    VLOG(1) << "Solver exited after only 1 iteration but passed "
               "backtracking checks, which may indicate an almost "
               "converged initial operating point and strategies.";
    CHECK_LT(
        (initial_operating_point.xs.back() - current_operating_point.xs.back())
            .cwiseAbs()
            .maxCoeff(),
        params_.convergence_tolerance);
  }

  CHECK(!player_costs_.front().AreConstraintsOn() ||
        was_operating_point_feasible);

  // Set final strategies and operating point.
  final_strategies->swap(current_strategies);
  final_operating_point->swap(current_operating_point);

  return true;
}

bool GameSolver::CurrentOperatingPoint(
    const OperatingPoint& last_operating_point,
    const std::vector<Strategy>& current_strategies,
    OperatingPoint* current_operating_point, bool* has_converged,
    std::vector<float>* total_costs, bool check_trust_region,
    bool* satisfies_constraints) const {
  CHECK_NOTNULL(current_operating_point);
  CHECK_NOTNULL(has_converged);
  CHECK_NOTNULL(total_costs);

  current_operating_point->t0 = last_operating_point.t0;
  *has_converged = true;
  if (satisfies_constraints) *satisfies_constraints = true;
  if (total_costs->size() != player_costs_.size())
    total_costs->resize(player_costs_.size());
  std::fill(total_costs->begin(), total_costs->end(), 0.0);

  // Integrate dynamics and populate operating point, one time step at a time.
  VectorXf x(last_operating_point.xs[0]);
  for (size_t kk = 0; kk < num_time_steps_; kk++) {
    const Time t = last_operating_point.t0 + ComputeTimeStamp(kk);

    // Unpack.
    const VectorXf delta_x = x - last_operating_point.xs[kk];
    const auto& last_us = last_operating_point.us[kk];
    auto& current_us = current_operating_point->us[kk];

    // Accumulate costs.
    for (size_t ii = 0; ii < player_costs_.size(); ii++)
      (*total_costs)[ii] += player_costs_[ii].Evaluate(t, x, current_us);

    // Check convergence and trust region (including explicit inequality
    // constraints).
    auto check_all_constraints = [this](Time t, const VectorXf& x,
                                        const std::vector<VectorXf>& us) {
      for (const auto& cost : this->player_costs_) {
        if (!cost.CheckConstraints(t, x, us)) return false;
      }
      return true;
    };  // check_all_constraints

    const float delta_x_distance = StateDistance(
        x, last_operating_point.xs[kk], params_.trust_region_dimensions);
    const bool checked_constraints =
        check_all_constraints(t, x, current_us) || !satisfies_constraints;

    *has_converged &= (delta_x_distance < params_.convergence_tolerance &&
                       checked_constraints);

    if (check_trust_region) {
      if (satisfies_constraints) *satisfies_constraints &= checked_constraints;

      if (delta_x_distance > params_.trust_region_size ||
          (player_costs_.front().AreConstraintsOn() && !checked_constraints)) {
        // If we still satisfy constraints then log a warning. This shouldn't
        // really ever lead to a fault though since the solver should be
        // backtracking if this returns false anyway.
        if (checked_constraints)
          VLOG(2) << "Failed trust region on time step " << kk
                  << " but satisfied constraints up till then.";
        return false;
      }
    }

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

  return true;
}

float GameSolver::StateDistance(const VectorXf& x1, const VectorXf& x2,
                                const std::vector<Dimension>& dims) const {
  if (dims.empty()) return (x1 - x2).cwiseAbs().maxCoeff();

  float distance = 0.0;
  for (const Dimension dim : dims) distance += std::abs(x1(dim) - x2(dim));

  return distance;
}

bool GameSolver::ModifyLQStrategies(std::vector<Strategy>* strategies,
                                    OperatingPoint* current_operating_point,
                                    bool* is_new_operating_point_feasible,
                                    bool* has_converged,
                                    std::vector<float>* total_costs) const {
  CHECK_NOTNULL(strategies);
  CHECK_NOTNULL(current_operating_point);
  CHECK_NOTNULL(has_converged);
  CHECK_NOTNULL(total_costs);

  // Initially scale alphas by a fixed amount to avoid unnecessary
  // backtracking.
  ScaleAlphas(params_.initial_alpha_scaling, strategies);

  // Compute next operating point.
  const OperatingPoint last_operating_point(*current_operating_point);
  bool satisfies_trust_region = CurrentOperatingPoint(
      last_operating_point, *strategies, current_operating_point, has_converged,
      total_costs, true, is_new_operating_point_feasible);

  if (!params_.linesearch) return true;

  // Keep reducing alphas until we satisfy the trust region constraint.
  for (size_t ii = 0; ii < params_.max_backtracking_steps; ii++) {
    if (satisfies_trust_region) return true;

    ScaleAlphas(params_.geometric_alpha_scaling, strategies);
    satisfies_trust_region = CurrentOperatingPoint(
        last_operating_point, *strategies, current_operating_point,
        has_converged, total_costs, true, is_new_operating_point_feasible);

    if (*has_converged && player_costs_.front().AreConstraintsOn())
      CHECK(*is_new_operating_point_feasible);
  }

  // Output a warning. Solver should revert to last valid operating point.
  VLOG(1) << "Exceeded maximum number of backtracking steps.";
  CHECK(!*has_converged);
  return false;
}

}  // namespace ilqgames
