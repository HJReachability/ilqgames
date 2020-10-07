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

#ifndef ILQGAMES_SOLVER_PROBLEM_H
#define ILQGAMES_SOLVER_PROBLEM_H

#include <ilqgames/cost/player_cost.h>
#include <ilqgames/dynamics/multi_player_dynamical_system.h>
#include <ilqgames/dynamics/multi_player_flat_system.h>
#include <ilqgames/dynamics/multi_player_integrable_system.h>
#include <ilqgames/utils/solver_log.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

#include <limits>
#include <memory>
#include <vector>

namespace ilqgames {

class Problem {
 public:
  virtual ~Problem() {}

  // Initialize this object.
  virtual void Initialize() {
    ConstructDynamics();
    ConstructPlayerCosts();
    ConstructInitialState();
    ConstructInitialOperatingPoint();
    ConstructInitialStrategies();
    initialized_ = true;
  }

  // Reset the initial time and change nothing else.
  void ResetInitialTime(Time t0) {
    CHECK(initialized_);
    operating_point_->t0 = t0;
  }

  void ResetInitialState(const VectorXf& x0) {
    CHECK(initialized_);
    x0_ = x0;
  }

  // Update initial state and modify previous strategies and operating
  // points to start at the specified runtime after the current time t0.
  // Since time is continuous and we will want to maintain the same fixed
  // discretization, we will integrate x0 forward from t0 by approximately
  // planner_runtime, then find the nearest state in the existing plan to that
  // state, and start from there. By default, extends operating points and
  // strategies as follows:
  // 1. new controls are zero
  // 2. new states are those that result from zero control
  // 3. new strategies are also zero
  virtual void SetUpNextRecedingHorizon(const VectorXf& x0, Time t0,
                                        Time planner_runtime = 0.1);

  // Overwrite existing solution with the given operating point and strategies.
  // Truncates to fit in the same memory.
  virtual void OverwriteSolution(const OperatingPoint& operating_point,
                                 const std::vector<Strategy>& strategies);

  // Accessors.
  bool IsConstrained() const;
  virtual Time InitialTime() const { return operating_point_->t0; }
  const VectorXf& InitialState() const { return x0_; }
  // size_t NumTimeSteps() const { return num_time_steps_; }
  // Time TimeStep() const { return time_step_; }
  // Time TimeHorizon() const { return time_horizon_; }
  std::vector<PlayerCost>& PlayerCosts() { return player_costs_; }
  const std::vector<PlayerCost>& PlayerCosts() const { return player_costs_; }
  const std::shared_ptr<const MultiPlayerIntegrableSystem>& Dynamics() const {
    return dynamics_;
  }
  const MultiPlayerDynamicalSystem& NormalDynamics() const {
    CHECK(!dynamics_->TreatAsLinear());
    return *static_cast<const MultiPlayerDynamicalSystem*>(dynamics_.get());
  }
  const MultiPlayerFlatSystem& FlatDynamics() const {
    CHECK(dynamics_->TreatAsLinear());
    return *static_cast<const MultiPlayerFlatSystem*>(dynamics_.get());
  }
  virtual const OperatingPoint& CurrentOperatingPoint() const {
    return *operating_point_;
  }
  virtual const std::vector<Strategy>& CurrentStrategies() const {
    return *strategies_;
  }

 protected:
  Problem();

  // Functions for initialization. By default, operating point and strategies
  // are initialized to zero.
  virtual void ConstructDynamics() = 0;
  virtual void ConstructPlayerCosts() = 0;
  virtual void ConstructInitialState() = 0;
  virtual void ConstructInitialOperatingPoint() {
    operating_point_.reset(
        new OperatingPoint(time::kNumTimeSteps, 0.0, dynamics_));
  }
  virtual void ConstructInitialStrategies() {
    strategies_.reset(new std::vector<Strategy>());
    for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++)
      strategies_->emplace_back(time::kNumTimeSteps, dynamics_->XDim(),
                                dynamics_->UDim(ii));
  }

  // Utility used by SetUpNextRecedingHorizon. Integrate the given state
  // forward, set the new initial state and time, and return the first timestep
  // in the new problem.
  size_t SyncToExistingProblem(const VectorXf& x0, Time t0,
                               Time planner_runtime, OperatingPoint& op);

  // // Time horizon (s), time step (s), and number of time steps.
  // const Time time_horizon_;
  // const Time time_step_;
  // const size_t num_time_steps_;

  // Dynamical system.
  std::shared_ptr<const MultiPlayerIntegrableSystem> dynamics_;

  // Player costs. These will not change during operation of this solver.
  std::vector<PlayerCost> player_costs_;

  // Initial condition.
  VectorXf x0_;

  // Strategies and operating points for all players.
  std::unique_ptr<OperatingPoint> operating_point_;
  std::unique_ptr<std::vector<Strategy>> strategies_;

  // Has this object been initialized?
  bool initialized_;
};  // class Problem

}  // namespace ilqgames

#endif
