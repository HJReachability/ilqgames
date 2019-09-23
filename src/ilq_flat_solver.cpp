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

bool ILQFlatSolver::Solve(const VectorXf& xi0,
                          const OperatingPoint& initial_operating_point,
                          const std::vector<Strategy>& initial_strategies,
                          OperatingPoint* final_operating_point,
                          std::vector<Strategy>* final_strategies,
                          SolverLog* log, Time max_runtime) {
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

  // Cast dynamics.
  const auto dyn = static_cast<const MultiPlayerFlatSystem*>(dynamics_.get());

  // Flag for whether or not the initial operating point is all zeros.
  // If it is all zeros, we will populate it with the initial state unrolled
  // with the initial strategies in the first iteration of the solver, but
  // otherwise we will leave it alone.
  const bool is_initial_operating_point_zero =
      initial_operating_point.xs[0].squaredNorm() < constants::kSmallNumber;

  // Last and current operating points.
  // OperatingPoint last_operating_point(num_time_steps_, dyn->NumPlayers(),
  //                                     initial_operating_point.t0);
  OperatingPoint last_operating_point(initial_operating_point);
  OperatingPoint current_operating_point(initial_operating_point);

  // Ensure that the current operating point starts at the initial state.
  current_operating_point.xs[0] = xi0;
  last_operating_point.xs[0] = xi0;

  // Current strategies.
  std::vector<Strategy> current_strategies(initial_strategies);

  // Preallocate vectors for linearizations and quadraticizations.
  // Both are time-indexed (and quadraticizations' inner vector is indexed by
  // player).
  std::vector<LinearDynamicsApproximation> linearization(
      num_time_steps_, dyn->LinearizedSystem());
  std::vector<std::vector<QuadraticCostApproximation>> quadraticization(
      num_time_steps_);
  for (auto& quads : quadraticization)
    quads.resize(dyn->NumPlayers(), QuadraticCostApproximation(dyn->XDim()));

  // // Preallocate vector of non-linear system controls.
  // std::vector<VectorXf> us(dyn->NumPlayers());
  // for (PlayerIndex ii = 0; ii < dyn->NumPlayers(); ii++)
  //   us[ii].resize(dyn->UDim(ii));

  // Number of iterations, starting from 0.
  size_t num_iterations = 0;

  // Log initial iterate.
  if (log) {
    log->AddSolverIterate(current_operating_point, current_strategies,
                          EvaluateCosts(current_operating_point),
                          elapsed_time(solver_call_time));
  }

  // Keep iterating until convergence.
  constexpr Time kMaxIterationRuntimeGuess = 2e-2;  // s
  while (elapsed_time(solver_call_time) <
             max_runtime - kMaxIterationRuntimeGuess &&
         !HasConverged(num_iterations, last_operating_point,
                       current_operating_point)) {
    // New iteration.
    num_iterations++;

    // auto print_ops = [&last_operating_point,
    //                   &current_operating_point](const std::string& msg) {
    //   std::cout << msg << std::endl;
    //   for (size_t kk = 0; kk < last_operating_point.xs.size(); kk++) {
    //     std::cout << kk
    //               << " last x = " << last_operating_point.xs[kk].transpose()
    //               << "\n current x = "
    //               << current_operating_point.xs[kk].transpose() << std::endl;
    //   }
    // };

    // Swap operating points and compute new current operating point.
    if (num_iterations > 1 || is_initial_operating_point_zero) {
      last_operating_point.swap(current_operating_point);
      CurrentOperatingPoint(last_operating_point, current_strategies,
                            &current_operating_point);
    }

    // Linearize dynamics and quadraticize costs for all players about the new
    // operating point.
    for (size_t kk = 0; kk < num_time_steps_; kk++) {
      const Time t = initial_operating_point.t0 + ComputeTimeStamp(kk);
      const auto& xi = current_operating_point.xs[kk];
      const auto& vs = current_operating_point.us[kk];

      // const VectorXf x = dyn->FromLinearSystemState(xi);
      // us = dyn->LinearizingControls(x, vs);

      // Quadraticize costs.
      std::transform(player_costs_.begin(), player_costs_.end(),
                     quadraticization[kk].begin(),
                     [&t, &xi, &vs](const PlayerCost& cost) {
                       return cost.Quadraticize(t, xi, vs);
                     });

      // dyn->ChangeControlCostCoordinates(xi, &quadraticization[kk]);
    }

    // Solve LQ game.
    current_strategies =
        SolveLQGame(*dynamics_, linearization, quadraticization);

    // Modify this LQ solution.
    if (!ModifyLQStrategies(current_operating_point, &current_strategies))
      return false;

    // Log current iterate.
    if (log) {
      log->AddSolverIterate(current_operating_point, current_strategies,
                            EvaluateCosts(current_operating_point),
                            elapsed_time(solver_call_time));
    }
  }

  // Set final strategies and operating point.
  final_strategies->swap(current_strategies);
  final_operating_point->swap(current_operating_point);

  return true;
}

bool ILQFlatSolver::SatisfiesTrustRegion(
    const OperatingPoint& last_operating_point,
    const OperatingPoint& current_operating_point) const {
  // Check if all states are far from singularity.
  const auto& dyn = *static_cast<const MultiPlayerFlatSystem*>(dynamics_.get());
  for (size_t ii = 0; ii < current_operating_point.xs.size(); ii++) {
    if (dyn.IsLinearSystemStateSingular(current_operating_point.xs[ii])) {
      return false;
    }
  }

  // Are the operating points close.
  return AreOperatingPointsClose(last_operating_point, current_operating_point,
                                 params_.trust_region_size,
                                 params_.trust_region_dimensions);
}

bool ILQFlatSolver::AreOperatingPointsClose(
    const OperatingPoint& op1, const OperatingPoint& op2, float threshold,
    const std::vector<Dimension>& dims) const {
  CHECK_EQ(op1.xs.size(), op2.xs.size());
  const auto& dyn = *static_cast<const MultiPlayerFlatSystem*>(dynamics_.get());

  for (size_t kk = 0; kk < op1.xs.size(); kk++) {
    VectorXf x1 = op1.xs[kk];
    VectorXf x2 = op2.xs[kk];

    // If not singular, use nonlinear system states.
    if (!dyn.IsLinearSystemStateSingular(x1))
      x1 = dyn.FromLinearSystemState(x1);
    if (!dyn.IsLinearSystemStateSingular(x2))
      x2 = dyn.FromLinearSystemState(x2);

    if (dims.empty() && (x1 - x2).cwiseAbs().maxCoeff() > threshold)
      return false;
    else if (!dims.empty()) {
      for (const Dimension dim : dims) {
        if (std::abs(x1(dim) - x2(dim)) > threshold) return false;
      }
    }
  }

  return true;
}

}  // namespace ilqgames
