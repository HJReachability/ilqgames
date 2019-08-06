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

#ifndef ILQGAMES_SOLVER_ILQ_SOLVER_H
#define ILQGAMES_SOLVER_ILQ_SOLVER_H

#include <ilqgames/cost/player_cost.h>
#include <ilqgames/dynamics/multi_player_dynamical_system.h>
#include <ilqgames/utils/linear_dynamics_approximation.h>
#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/quadratic_cost_approximation.h>
#include <ilqgames/utils/solver_log.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <limits>
#include <memory>
#include <vector>

namespace ilqgames {

class ILQSolver {
 public:
  virtual ~ILQSolver() {}
  ILQSolver(const std::shared_ptr<const MultiPlayerDynamicalSystem>& dynamics,
            const std::vector<PlayerCost>& player_costs, Time time_horizon,
            Time time_step)
      : dynamics_(dynamics),
        player_costs_(player_costs),
        time_horizon_(time_horizon),
        time_step_(time_step),
        num_time_steps_(static_cast<size_t>(time_horizon / time_step)) {
    CHECK_EQ(player_costs_.size(), dynamics_->NumPlayers());
    CHECK_NOTNULL(dynamics_.get());
  }

  // Solve this game. Returns true if converged.
  bool Solve(const VectorXf& x0, const OperatingPoint& initial_operating_point,
             const std::vector<Strategy>& initial_strategies,
             OperatingPoint* final_operating_point,
             std::vector<Strategy>* final_strategies, SolverLog* log = nullptr,
             Time max_runtime = std::numeric_limits<Time>::infinity());

  // Accessors.
  Time TimeHorizon() const { return time_horizon_; }
  Time TimeStep() const { return time_step_; }
  const std::vector<PlayerCost>& PlayerCosts() const { return player_costs_; }
  const MultiPlayerDynamicalSystem& Dynamics() const { return *dynamics_; }

  // Compute time stamp from time index.
  Time ComputeTimeStamp(size_t time_index) const {
    return time_step_ * static_cast<Time>(time_index);
  }

 protected:
  // Modify LQ strategies to improve convergence properties.
  // This function replaces an Armijo linesearch that would take place in ILQR.
  // Returns true if successful.
  virtual bool ModifyLQStrategies(const OperatingPoint& current_operating_point,
                                  std::vector<Strategy>* strategies) const;

  // Check convergence. Returns true if converged.
  virtual bool HasConverged(
      size_t iteration, const OperatingPoint& last_operating_point,
      const OperatingPoint& current_operating_point) const;

  // Compute the current operating point based on the current set of strategies
  // and the last operating point.
  void CurrentOperatingPoint(const OperatingPoint& last_operating_point,
                             const std::vector<Strategy>& current_strategies,
                             OperatingPoint* current_operating_point) const;

  // Dynamical system.
  const std::shared_ptr<const MultiPlayerDynamicalSystem> dynamics_;

  // Player costs. These will not change during operation of this solver.
  const std::vector<PlayerCost> player_costs_;

  // Time horizon (s), time step (s), and number of time steps.
  const Time time_horizon_;
  const Time time_step_;
  const size_t num_time_steps_;
};  // class ILQSolver

}  // namespace ilqgames

#endif
