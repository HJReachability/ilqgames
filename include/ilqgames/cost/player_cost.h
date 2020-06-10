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
// Container to store all the cost functions for a single player, and keep track
// of which variables (x, u1, u2, ..., uN) they correspond to.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_COST_PLAYER_COST_H
#define ILQGAMES_COST_PLAYER_COST_H

#include <ilqgames/constraint/constraint.h>
#include <ilqgames/cost/cost.h>
#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/quadratic_cost_approximation.h>
#include <ilqgames/utils/types.h>

#include <unordered_map>

namespace ilqgames {

class PlayerCost {
 public:
  ~PlayerCost() {}
  explicit PlayerCost(const std::string& name = "",
                      float state_regularization = 0.0,
                      float control_regularization = 0.0)
      : name_(name),
        state_regularization_(state_regularization),
        control_regularization_(control_regularization),
        are_constraints_on_(true) {}

  // Add new state and control costs for this player.
  void AddStateCost(const std::shared_ptr<Cost>& cost);
  void AddControlCost(PlayerIndex idx, const std::shared_ptr<Cost>& cost);

  // Add new state and control constraints for this player.
  void AddStateConstraint(const std::shared_ptr<Constraint>& constraint);
  void AddControlConstraint(PlayerIndex idx,
                            const std::shared_ptr<Constraint>& constraint);

  // Evaluate this cost at the current time, state, and controls, or integrate
  // over an entire trajectory. Does *not* incorporate cost barriers due to
  // inequality constraints. The "Offset" here indicates that state costs will
  // be evaluated at the next time step.
  float Evaluate(Time t, const VectorXf& x,
                 const std::vector<VectorXf>& us) const;
  float Evaluate(const OperatingPoint& op, Time time_step) const;
  float EvaluateOffset(Time t, Time next_t, const VectorXf& next_x,
                       const std::vector<VectorXf>& us) const;

  // Quadraticize this cost at the given time, state, and controls.
  // *Does* account for cost barriers due to inequality constraints.
  QuadraticCostApproximation Quadraticize(
      Time t, const VectorXf& x, const std::vector<VectorXf>& us) const;

  // Turn all constraints either "on" or "off" (in which case they are replaced
  // with their"equivalent" costs).
  void TurnConstraintsOn() { are_constraints_on_ = true; }
  void TurnConstraintsOff() { are_constraints_on_ = false; }
  bool AreConstraintsOn() const { return are_constraints_on_; }

  // Check whether constraints are satisfied at the given time.
  bool CheckConstraints(Time t, const VectorXf& x,
                        const std::vector<VectorXf>& us) const;
  size_t NumStateConstraints() const { return state_constraints_.size(); }
  size_t NumControlConstraints() const { return control_constraints_.size(); }

  // Scale all weights associated with all constraint barriers by the given
  // multiplier, which ought to be less than 1.0. Can also reset all weights
  // to 1.0.
  void ScaleConstraintBarrierWeights(float scale = 0.5);
  void ResetConstraintBarrierWeights();

  // Set exponential constant for all costs associated to this player.
  void SetExponentialConstant(float a);

  // Accessors.
  const std::vector<std::shared_ptr<Cost>>& StateCosts() const {
    return state_costs_;
  }
  const CostMap<Cost>& ControlCosts() const { return control_costs_; }
  const std::vector<std::shared_ptr<Constraint>>& StateConstraints() const {
    return state_constraints_;
  }
  const CostMap<Constraint>& ControlConstraints() const {
    return control_constraints_;
  }

 private:
  // Name to be used with error msgs.
  const std::string name_;

  // State costs and control costs.
  std::vector<std::shared_ptr<Cost>> state_costs_;
  CostMap<Cost> control_costs_;

  // State and control constraints. Control constraints can apply to any
  // player's control input, though it likely only makes sense to apply them to
  // this player's input.
  std::vector<std::shared_ptr<Constraint>> state_constraints_;
  CostMap<Constraint> control_constraints_;
  bool are_constraints_on_;

  // Regularization on costs.
  const float state_regularization_, control_regularization_;
};  //\class PlayerCost

}  // namespace ilqgames

#endif
