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
// Base class for all iterative LQ game solvers. Derives from FeedbackSolver.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_SOLVER_ILQ_SOLVER_H
#define ILQGAMES_SOLVER_ILQ_SOLVER_H

#include <ilqgames/cost/player_cost.h>
#include <ilqgames/dynamics/multi_player_dynamical_system.h>
#include <ilqgames/solver/game_solver.h>
#include <ilqgames/solver/lq_feedback_solver.h>
#include <ilqgames/solver/lq_solver.h>
#include <ilqgames/solver/solver_params.h>
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

class ILQSolver : public GameSolver {
public:
  virtual ~ILQSolver() {}
  ILQSolver(const std::shared_ptr<Problem> &problem,
            const SolverParams &params = SolverParams())
      : GameSolver(problem, params), linearization_(time::kNumTimeSteps),
        cost_quadraticization_(time::kNumTimeSteps),
        last_merit_function_value_(constants::kInfinity),
        expected_decrease_(constants::kInfinity) {
    // Set up LQ solver.
    if (params_.open_loop)
      lq_solver_.reset(
          new LQOpenLoopSolver(problem_->Dynamics(), time::kNumTimeSteps));
    else
      lq_solver_.reset(
          new LQFeedbackSolver(problem_->Dynamics(), time::kNumTimeSteps));

    // If this system is flat then compute the linearization once, now.
    if (problem_->Dynamics()->TreatAsLinear())
      ComputeLinearization(&linearization_);

    // Prepopulate quadraticization.
    for (auto &quads : cost_quadraticization_)
      quads.resize(problem_->Dynamics()->NumPlayers(),
                   QuadraticCostApproximation(problem_->Dynamics()->XDim()));

    // Set last quadraticization to current, to start.
    last_cost_quadraticization_ = cost_quadraticization_;
  }

  // Solve this game. Returns true if converged.
  virtual std::shared_ptr<SolverLog>
  Solve(bool *success = nullptr,
        Time max_runtime = std::numeric_limits<Time>::infinity());

  // Accessors.
  // NOTE: these should be primarily used by higher-level solvers.
  std::vector<std::vector<QuadraticCostApproximation>> *Quadraticization() {
    return &cost_quadraticization_;
  }

  std::vector<LinearDynamicsApproximation> *Linearization() {
    return &linearization_;
  }

protected:
  // Modify LQ strategies to improve convergence properties.
  // This function performs an Armijo linesearch and returns true if successful.
  bool ModifyLQStrategies(const std::vector<VectorXf> &delta_xs,
                          const std::vector<std::vector<VectorXf>> &costates,
                          std::vector<Strategy> *strategies,
                          OperatingPoint *current_operating_point,
                          bool *has_converged);

  // Compute distance (infinity norm) between states in the given dimensions.
  // If dimensions empty, checks all dimensions.
  float StateDistance(const VectorXf &x1, const VectorXf &x2,
                      const std::vector<Dimension> &dims) const;

  // Check if solver has converged.
  virtual bool HasConverged(float current_merit_function_value) const {
    return (current_merit_function_value <= last_merit_function_value_) &&
           std::abs(last_merit_function_value_ - current_merit_function_value) <
               params_.convergence_tolerance;
  }

  // Compute overall costs and set times of extreme costs.
  void TotalCosts(const OperatingPoint &current_op,
                  std::vector<float> *total_costs) const;

  // Armijo condition check. Returns true if the new operating point satisfies
  // the Armijo condition, and also returns current merit function value.
  bool CheckArmijoCondition(float current_merit_function_value,
                            float current_stepsize) const;

  // Compute current merit function value. Note that to compute the merit
  // function at the given operating point we have to compute a full cost
  // quadraticization there. To do so efficiently, this will overwrite the
  // current cost quadraticization (and presume it has already been used to
  // compute the expected decrease from the last iterate).
  float MeritFunction(const OperatingPoint &current_op,
                      const std::vector<std::vector<VectorXf>> &costates);

  // Compute expected decrease based on current cost quadraticization,
  // (player-indexed) strategies, and (time-indexed) lists of delta states and
  // (also player-indexed) costates.
  float
  ExpectedDecrease(const std::vector<Strategy> &strategies,
                   const std::vector<VectorXf> &delta_xs,
                   const std::vector<std::vector<VectorXf>> &costates) const;

  // Compute the current operating point based on the current set of
  // strategies and the last operating point.
  void CurrentOperatingPoint(const OperatingPoint &last_operating_point,
                             const std::vector<Strategy> &current_strategies,
                             OperatingPoint *current_operating_point) const;

  // Populate the given vector with a linearization of the dynamics about
  // the given operating point. Provide version with no operating point for use
  // with feedback linearizable systems.
  void
  ComputeLinearization(const OperatingPoint &op,
                       std::vector<LinearDynamicsApproximation> *linearization);
  void
  ComputeLinearization(std::vector<LinearDynamicsApproximation> *linearization);

  // Compute the quadratic cost approximation at the given operating point.
  void ComputeCostQuadraticization(
      const OperatingPoint &op,
      std::vector<std::vector<QuadraticCostApproximation>> *q);

  // Linearization and quadraticization. Both are time-indexed (and
  // quadraticizations' inner vector is indexed by player). Also keep track of
  // the quadraticization from last iteration.
  std::vector<LinearDynamicsApproximation> linearization_;
  std::vector<std::vector<QuadraticCostApproximation>> cost_quadraticization_;
  std::vector<std::vector<QuadraticCostApproximation>>
      last_cost_quadraticization_;

  // Core LQ Solver.
  std::unique_ptr<LQSolver> lq_solver_;

  // Last merit function value and expected decreases (per step length).
  float last_merit_function_value_;
  float expected_decrease_;
}; // class ILQSolver

} // namespace ilqgames

#endif
