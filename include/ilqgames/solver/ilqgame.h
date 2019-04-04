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

#ifndef ILQGAMES_SOLVER_ILQGAME_H
#define ILQGAMES_SOLVER_ILQGAME_H

#include <ilqgames/cost/player_cost.h>
#include <ilqgames/utils/types.h>

#include <memory>
#include <vector>

namespace ilqgames {

template <typename DynamicsType>
class ILQGame {
 public:
  virtual ~ILQGame() {}

  // Solve this game. Returns true if converged.
  bool Solve(const VectorXf& x0,
             const std::vector<Strategy>& initial_strategies,
             std::vector<Strategy>* final_strategies);

 protected:
  explicit ILQGame(const DynamicsType& dynamics,
                   const std::vector<PlayerCost>& player_costs,
                   Time time_horizon, Time time_step)
      : dynamics_(dynamics),
        player_costs_(player_costs),
        time_horizon_(time_horizon),
        time_step_(time_step) {}

  // Compute the current operating point based on the current set of strategies.
  OperatingPoint ComputeOperatingPoint(
      const VectorXf& x0, const std::vector<Strategy>& strategies) const;

  // Modify LQ strategies to improve convergence properties.
  // This function replaces an Armijo linesearch that would take place in ILQR.
  // Returns true if successful.
  virtual bool ModifyLQStrategies(
      const std::vector<Strategy>& current_strategies,
      std::vector<Strategy>* next_strategies) const = 0;

  // Check convergence. Returns true if converged.
  virtual bool HasConverged(
      size_t iteration, const std::vector<Strategy>& current_strategies,
      const std::vector<Strategy>& next_strategies) const = 0;

  // Dynamical system.
  const DynamicsType dynamics_;

  // Player costs. These will not change during operation of this solver.
  const std::vector<PlayerCost> player_costs_;

  // Time horizon (s), time step (s), and number of time steps.
  const Time time_horizon_;
  const Time time_step_;
  const size_t num_time_steps_;
};  //\class ILQGame

// ----------------------------- IMPLEMENTATION ----------------------------- //

template <typename DynamicsType>
bool ILQGame<DynamicsType>::Solve(
    const VectorXf& x0, const std::vector<Strategy>& initial_strategies,
    const std::vector<Strategy>* final_strategies) {
  // Make sure we have enough strategies for each time step.
  DCHECK_EQ(dynamics_.NumPlayers(), initial_strategies.size());
  DCHECK(std::accumulate(
      initial_strategies.begin(), initial_strategies.end(), true,
      [&num_time_steps_](bool correct_so_far, const Strategy& strategy) {
        return correct_so_far &= strategy.Ps.size() == num_time_steps_ &&
                                 strategy.alphas.size() == num_time_steps_;
      }));

  // Current and next strategies.
  std::vector<Strategy> current_strategies(initial_strategies);
  std::vector<Strategy> next_strategies(dynamics_.NumPlayers());

  // Preallocate vectors for linearizations and quadraticizations.
  // Both are time-indexed (and quadraticizations' inner vector is indexed by
  // player).
  std::vector<LinearDynamicsApproximation> linearization(num_time_steps_);
  std::vector<std::vector<QuadraticCostApproximation>> quadraticization(
      num_time_steps_);
  for (auto& quads : quadraticization) quads.reserve(dynamics_.NumPlayers());

  // Number of iterations, starting from 0.
  size_t num_iterations = 0;

  // Keep iterating until convergence.
  while (!HasConverged(num_iterations, current_strategies, next_strategies)) {
    num_iterations++;

    // Update current and next strategies. A simple swap will work here.
    current_strategies.swap(next_strategies);

    // Compute the current operating point.
    const OperatingPoint operating_point =
        ComputeOperatingPoint(x0, current_strategies);

    // Helper function to compute time stamp from time index.
    auto compute_time_stamp = [&time_step_](size_t time_index) {
      return time_step_ * static_cast<Time>(time_index);
    };  // compute_time_stamp

    // Linearize dynamics and quadraticize costs for all players about the new
    // operating point.
    for (size_t ii = 0; ii < num_time_steps_; ii++) {
      const Time t = compute_time_stamp(ii);
      const auto& x = operating_point.xs[ii];
      const auto& us = operating_points.us[ii];

      // Linearize dynamics.
      linearization[ii] = dynamics_.Linearize(t, x, us);

      // Quadraticize costs.
      std::transform(player_costs_.begin(), player_costs_.end(),
                     quadraticizations[ii].begin(),
                     [&t, &x, &us](const PlayerCost& cost) {
                       return cost.Quadraticize(t, x, us)
                     });
    }

    // Solve LQ game.
    next_strategies = SolveLQGame(linearization, quadraticization);

    // Modify this LQ solution.
    ModifyLQStrategies(current_strategies, &next_strategies);
  }
}

template <typename DynamicsType>
OperatingPoint ILQGame<DynamicsType>::ComputeOperatingPoint(
    const VectorXf& x0, const std::vector<Strategy>& strategies) const {
  // TODO!
}

}  // namespace ilqgames

#endif
