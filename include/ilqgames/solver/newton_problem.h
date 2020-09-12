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
// Newton problem, which derives from Problem but makes all operating points and
// strategies references to a single primal vector, and also has an accompanying
// dual vector which is referenced in each individual multiplier.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_SOLVER_NEWTON_PROBLEM_H
#define ILQGAMES_SOLVER_NEWTON_PROBLEM_H

#include <ilqgames/cost/player_cost.h>
#include <ilqgames/dynamics/multi_player_integrable_system.h>
#include <ilqgames/solver/problem.h>
#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/solver_log.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

#include <limits>
#include <memory>
#include <vector>

namespace ilqgames {

class NewtonProblem : public Problem {
 public:
  virtual ~NewtonProblem() {}

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

  // Compute the number of primal and dual variables in this problem.
  size_t NumVariables() const { return NumPrimals() + NumDuals(); }
  virtual size_t NumPrimals() const;
  virtual size_t NumDuals() const;
  size_t NumOperatingPointVariables() const;
  virtual size_t KKTSystemSize() const;

 protected:
  NewtonProblem() : Problem() {}

  // Initialize this object.
  virtual void Initialize() {
    ConstructDynamics();
    ConstructPlayerCosts();
    ConstructInitialState();
    ConstructPrimalsAndDuals();
    initialized_ = true;
  }

  // Functions for initialization. By default, primals and duals are initialized
  // to zero.
  virtual void ConstructDynamics() = 0;
  virtual void ConstructPlayerCosts() = 0;
  virtual void ConstructInitialState() = 0;
  virtual void ConstructPrimalsAndDuals();
  virtual void ConstructInitialOperatingPoint();
  virtual void ConstructInitialStrategies();
  virtual void ConstructInitialLambdas();

  // Primal variables.
  VectorXf primals_;
  std::unique_ptr<OperatingPointRef> operating_point_ref_;
  std::unique_ptr<std::vector<StrategyRef>> strategy_refs_;

  // Dual variables.
  VectorXf duals_;
  std::unique_ptr<std::vector<RefVector>> lambda_dyns_;
  std::unique_ptr<std::vector<RefVector>> lambda_feedbacks_;
  std::unique_ptr<std::vector<std::vector<RefVector>>>
      lambda_state_constraints_;
  std::unique_ptr<std::vector<std::vector<PlayerDualMap>>>
      lambda_control_constraints_;
};  // class NewtonProblem

}  // namespace ilqgames

#endif
