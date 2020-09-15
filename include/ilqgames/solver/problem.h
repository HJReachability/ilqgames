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

  // Compute time stamp from time index.
  Time ComputeRelativeTimeStamp(size_t time_index) const {
    return time_step_ * static_cast<Time>(time_index);
  }

  // Accessors.
  virtual Time InitialTime() const { return operating_point_->t0; }
  const VectorXf& InitialState() const { return x0_; }
  size_t NumTimeSteps() const { return num_time_steps_; }
  Time TimeStep() const { return time_step_; }
  Time TimeHorizon() const { return time_horizon_; }
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
  const OperatingPoint& CurrentOperatingPoint() const {
    return *operating_point_;
  }
  const std::vector<Strategy>& CurrentStrategies() const {
    return *strategies_;
  }

 protected:
  Problem();

  // Initialize this object.
  virtual void Initialize() {
    ConstructDynamics();
    ConstructPlayerCosts();
    ConstructInitialState();
    ConstructInitialOperatingPoint();
    ConstructInitialStrategies();
    initialized_ = true;
  }

  // Functions for initialization. By default, operating point and strategies
  // are initialized to zero.
  virtual void ConstructDynamics() = 0;
  virtual void ConstructPlayerCosts() = 0;
  virtual void ConstructInitialState() = 0;
  virtual void ConstructInitialOperatingPoint() {
    operating_point_.reset(new OperatingPoint(num_time_steps_, 0.0, dynamics_));
  }
  virtual void ConstructInitialStrategies() {
    strategies_.reset(new std::vector<Strategy>());
    for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++)
      strategies_->emplace_back(num_time_steps_, dynamics_->XDim(),
                                dynamics_->UDim(ii));
  }

  // Utility used by SetUpNextRecedingHorizon. Integrate the given state
  // forward, set the new initial state and time, and return the first timestep
  // in the new problem. Templated to handle both OperatingPoint and
  // OperatingPointRef.
  template <typename T>
  size_t SyncToExistingProblem(const VectorXf& x0, Time t0,
                               Time planner_runtime, T& op);

  // Time horizon (s), time step (s), and number of time steps.
  const Time time_horizon_;
  const Time time_step_;
  const size_t num_time_steps_;

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
};  // class NewtonProblem

// --------------------------- IMPLEMENTATION ------------------------------ //

template <typename T>
size_t Problem::SyncToExistingProblem(const VectorXf& x0, Time t0,
                                      Time planner_runtime, T& op) {
  CHECK(initialized_);
  CHECK_GE(planner_runtime, 0.0);
  CHECK_LE(planner_runtime + t0, InitialTime() + TimeHorizon());
  CHECK_GE(t0, operating_point_->t0);

  // Integrate x0 forward from t0 by approximately planner_runtime to get
  // actual initial state. Integrate up to the next discrete timestep, then
  // integrate for an integer number of discrete timesteps until by the *next*
  // timestep at least 'planner_runtime' has elapsed (done by rounding).
  constexpr float kRoundingError = 0.9;
  const Time relative_t0 = t0 - op.t0;
  size_t current_timestep = static_cast<size_t>(relative_t0 / time_step_);
  Time remaining_time_this_step =
      (current_timestep + 1) * time_step_ - relative_t0;
  if (remaining_time_this_step < kRoundingError * time_step_) {
    current_timestep += 1;
    remaining_time_this_step = time_step_ - remaining_time_this_step;
  }

  CHECK_LT(remaining_time_this_step, time_step_);

  // Initially, set x to the integrated version of x0 at the next timestep.
  VectorXf x = dynamics_->IntegrateToNextTimeStep(t0, x0, *operating_point_,
                                                  *strategies_);
  op.t0 = t0 + remaining_time_this_step;
  if (remaining_time_this_step <= planner_runtime) {
    const size_t num_steps_to_integrate = static_cast<size_t>(
        constants::kSmallNumber +  // Add to avoid truncation error.
        (planner_runtime - remaining_time_this_step) / time_step_);
    const size_t last_integration_timestep =
        current_timestep + num_steps_to_integrate;

    x = dynamics_->Integrate(current_timestep + 1, last_integration_timestep, x,
                             *operating_point_, *strategies_);
    op.t0 += time_step_ * num_steps_to_integrate;
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
  CHECK_LE(std::abs(t0 + planner_runtime - op.t0), time_step_);
  return first_timestep_in_new_problem;
}

}  // namespace ilqgames

#endif
