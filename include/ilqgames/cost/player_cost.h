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

#include <ilqgames/constraint/barrier/barrier.h>
#include <ilqgames/constraint/equality_constraint.h>
#include <ilqgames/cost/cost.h>
#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/quadratic_cost_approximation.h>
#include <ilqgames/utils/types.h>

#include <unordered_map>

namespace ilqgames {

class PlayerCost {
 public:
  ~PlayerCost() {}

  // Provide default values for all constructor values. If num_time_steps is
  // positive use that to initialize the lambdas.
  explicit PlayerCost(const std::string& name = "",
                      float state_regularization = 0.0,
                      float control_regularization = 0.0,
                      size_t num_time_steps = 0)
      : name_(name),
        state_regularization_(state_regularization),
        control_regularization_(control_regularization),
        are_barriers_on_(true),
        cost_structure_(CostStructure::SUM),
        time_of_extreme_cost_(0) {
    if (num_time_steps > 0) {
      state_lambdas_.resize(num_time_steps);
      control_lambdas_.resize(num_time_steps);
    }
  }

  // Add new state and control costs for this player.
  void AddStateCost(const std::shared_ptr<Cost>& cost);
  void AddControlCost(PlayerIndex idx, const std::shared_ptr<Cost>& cost);

  // Add new state and control barriers for this player.
  void AddStateBarrier(const std::shared_ptr<Barrier>& barrier);
  void AddControlBarrier(PlayerIndex idx,
                         const std::shared_ptr<Barrier>& barrier);

  // Add new state and control constraints. For now, they are only equality
  // constraints but later they should really be inequality constraints and
  // there should be some logic for maintaining sets of active constraints.
  void AddStateConstraint(
      const std::shared_ptr<EqualityConstraint>& constraint);
  void AddControlConstraint(
      PlayerIndex idx, const std::shared_ptr<EqualityConstraint>& constraint);

  // Evaluate this cost at the current time, state, and controls, or
  // integrate over an entire trajectory. Does *not* incorporate cost
  // barriers due to inequality barriers. The "Offset" here indicates that
  // state costs will be evaluated at the next time step.
  float Evaluate(Time t, const VectorXf& x,
                 const std::vector<VectorXf>& us) const;
  float Evaluate(const OperatingPoint& op, Time time_step) const;
  float EvaluateOffset(Time t, Time next_t, const VectorXf& next_x,
                       const std::vector<VectorXf>& us) const;

  // Evaluate squared norm of all constraint violations.
  float SquaredConstraintViolation(const OperatingPoint& op,
                                   Time time_step) const;

  // Quadraticize this cost at the given time, time step, state, and controls.
  // *Does* account for cost barriers due to inequality barriers.
  QuadraticCostApproximation Quadraticize(
      Time t, size_t time_step, const VectorXf& x,
      const std::vector<VectorXf>& us) const;

  // Return empty cost quadraticization except for barriers and control costs.
  QuadraticCostApproximation QuadraticizeBarriersAndControlCosts(
      Time t, const VectorXf& x, const std::vector<VectorXf>& us) const;

  // Turn all barriers either "on" or "off" (in which case they are replaced
  // with their"equivalent" costs).
  void TurnBarriersOn() { are_barriers_on_ = true; }
  void TurnBarriersOff() { are_barriers_on_ = false; }
  bool AreBarriersOn() const { return are_barriers_on_; }

  // Check whether barriers are satisfied at the given time.
  bool CheckBarriers(Time t, const VectorXf& x,
                     const std::vector<VectorXf>& us) const;
  size_t NumStateBarriers() const { return state_barriers_.size(); }
  size_t NumControlBarriers() const { return control_barriers_.size(); }

  // Check whether constraints are satisfied.
  bool CheckConstraints(Time t, const VectorXf& x,
                        const std::vector<VectorXf>& us) const;
  size_t NumStateConstraints() const { return state_constraints_.size(); }
  size_t NumControlConstraints() const { return control_constraints_.size(); }

  // Scale all weights associated with all barrier barriers by the given
  // multiplier, which ought to be less than 1.0. Can also reset all weights
  // to 1.0.
  void ScaleBarrierWeights(float scale = 0.5);
  void ResetBarrierWeights();

  // Set whether this is a time-additive, max-over-time, or min-over-time cost.
  // At each specific time, all costs are accumulated with the given operation.
  enum CostStructure { SUM, MAX, MIN };
  void SetTimeAdditive() { cost_structure_ = SUM; }
  void SetMaxOverTime() { cost_structure_ = MAX; }
  void SetMinOverTime() { cost_structure_ = MIN; }
  bool IsTimeAdditive() const { return cost_structure_ == SUM; }
  bool IsMaxOverTime() const { return cost_structure_ == MAX; }
  bool IsMinOverTime() const { return cost_structure_ == MIN; }

  // Keep track of the time of extreme costs.
  size_t TimeOfExtremeCost() { return time_of_extreme_cost_; }
  void SetTimeOfExtremeCost(size_t kk) { time_of_extreme_cost_ = kk; }

  // Accessors.
  const PtrVector<Cost>& StateCosts() const { return state_costs_; }
  const PlayerPtrMultiMap<Cost>& ControlCosts() const { return control_costs_; }
  const PtrVector<Barrier>& StateBarriers() const { return state_barriers_; }
  const PlayerPtrMultiMap<Barrier>& ControlBarriers() const {
    return control_barriers_;
  }
  const PtrVector<EqualityConstraint>& StateConstraints() const {
    return state_constraints_;
  }
  const PlayerPtrMultiMap<EqualityConstraint>& ControlConstraints() const {
    return control_constraints_;
  }

  // Access multipliers at a given timestep. These are indexed by constraint.
  std::vector<float>& StateLambdas(size_t kk) { return state_lambdas_[kk]; }
  PlayerMultiMap<float>& ControlLambdas(size_t kk) {
    return control_lambdas_[kk];
  }
  float Mu() const { return mu_; }
  void SetMu(float mu) { mu_ = mu; }
  void ScaleMu(float scale) { mu_ *= scale; }

 private:
  // Name to be used with error msgs.
  const std::string name_;

  // State costs and control costs.
  PtrVector<Cost> state_costs_;
  PlayerPtrMultiMap<Cost> control_costs_;

  // State and control barriers. Control barriers can apply to any
  // player's control input, though it likely only makes sense to apply them to
  // this player's input.
  PtrVector<Barrier> state_barriers_;
  PlayerPtrMultiMap<Barrier> control_barriers_;
  bool are_barriers_on_;

  // State and control constraints, with multipliers and augmented multipliers
  // indexed by time, then by constraint. These are inserted so that, at each
  // timestep, iterators at equivalent positions correspond to equivalent
  // constraints.
  // NOTE: Multiplier naming from https://bjack205.github.io/assets/ALTRO.pdf.
  PtrVector<EqualityConstraint> state_constraints_;
  PlayerPtrMultiMap<EqualityConstraint> control_constraints_;
  std::vector<std::vector<float>> state_lambdas_;
  std::vector<PlayerMultiMap<float>> control_lambdas_;
  float mu_;

  // Regularization on costs.
  const float state_regularization_;
  const float control_regularization_;

  // Ternary variable whether this objective is time-additive, max-over-time, or
  // min-over-time.
  CostStructure cost_structure_;

  // Keep track of the time of extreme costs. This will depend upon the current
  // operating point, and it will only be meaningful if the cost structure is an
  // extremum over time.
  size_t time_of_extreme_cost_;
};  //\class PlayerCost

}  // namespace ilqgames

#endif
