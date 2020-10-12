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

#include <ilqgames/cost/cost.h>
#include <ilqgames/cost/player_cost.h>
#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/quadratic_cost_approximation.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <unordered_map>

namespace ilqgames {

namespace {

// Accumulate control costs and constraints into the given quadratic
// approximation.
template <typename T, typename F>
void AccumulateControlCostsBase(const PlayerPtrMultiMap<T>& costs, Time t,
                                const std::vector<VectorXf>& us,
                                float regularization,
                                QuadraticCostApproximation* q, F f) {
  size_t cost_idx = 0;
  for (const auto& pair : costs) {
    const PlayerIndex player = pair.first;
    const auto& cost = pair.second;

    // If we haven't seen this player yet, initialize R and r to zero.
    auto iter = q->control.find(player);
    if (iter == q->control.end()) {
      auto inserted_pair = q->control.emplace(
          player, SingleCostApproximation(us[player].size(), regularization));

      // Second element should be true because we definitely won't have any
      // key collisions.
      CHECK(inserted_pair.second);

      // Update iter to point to where the new R was inserted.
      iter = inserted_pair.first;
    }

    f(*cost, t, us[player], &(iter->second.hess), &(iter->second.grad));
    cost_idx++;
  }
}

void AccumulateControlCosts(const PlayerPtrMultiMap<Cost>& costs, Time t,
                            const std::vector<VectorXf>& us,
                            float regularization,
                            QuadraticCostApproximation* q) {
  auto f = [](const Cost& cost, Time t, const VectorXf& u, MatrixXf* hess,
              VectorXf* grad) { cost.Quadraticize(t, u, hess, grad); };
  AccumulateControlCostsBase(costs, t, us, regularization, q, f);
}

void AccumulateControlConstraints(
    const PlayerPtrMultiMap<Constraint>& constraints, Time t,
    const std::vector<VectorXf>& us, float regularization,
    QuadraticCostApproximation* q) {
  auto f = [](const Constraint& constraint, Time t, const VectorXf& u,
              MatrixXf* hess,
              VectorXf* grad) { constraint.Quadraticize(t, u, hess, grad); };
  AccumulateControlCostsBase(constraints, t, us, regularization, q, f);
}

}  // namespace

void PlayerCost::AddStateCost(const std::shared_ptr<Cost>& cost) {
  state_costs_.emplace_back(cost);
}

void PlayerCost::AddControlCost(PlayerIndex idx,
                                const std::shared_ptr<Cost>& cost) {
  control_costs_.emplace(idx, cost);
}

void PlayerCost::AddStateConstraint(
    const std::shared_ptr<Constraint>& constraint) {
  state_constraints_.emplace_back(constraint);
}

void PlayerCost::AddControlConstraint(
    PlayerIndex idx, const std::shared_ptr<Constraint>& constraint) {
  control_constraints_.emplace(idx, constraint);
}

float PlayerCost::Evaluate(Time t, const VectorXf& x,
                           const std::vector<VectorXf>& us) const {
  float total_cost = 0.0;

  // State costs.
  for (const auto& cost : state_costs_) total_cost += cost->Evaluate(t, x);

  // Control costs.
  for (const auto& pair : control_costs_) {
    const PlayerIndex& player = pair.first;
    const auto& cost = pair.second;

    total_cost += cost->Evaluate(t, us[player]);
  }

  return total_cost;
}

float PlayerCost::Evaluate(const OperatingPoint& op, Time time_step) const {
  float cost = 0.0;
  if (IsMinOverTime())
    cost = constants::kInfinity;
  else if (IsMaxOverTime())
    cost = -constants::kInfinity;

  for (size_t kk = 0; kk < op.xs.size(); kk++) {
    const Time t = op.t0 + time_step * static_cast<float>(kk);
    const float instantaneous_cost = Evaluate(t, op.xs[kk], op.us[kk]);

    if (IsTimeAdditive())
      cost += instantaneous_cost;
    else if (IsMinOverTime())
      cost = std::min(cost, instantaneous_cost);
    else
      cost = std::max(cost, instantaneous_cost);
  }

  return cost;
}

float PlayerCost::Evaluate(const OperatingPoint& op) const {
  float total_cost = 0.0;
  for (size_t kk = 0; kk < op.xs.size(); kk++) total_cost += Evaluate(op, kk);

  return total_cost;
}

float PlayerCost::EvaluateOffset(Time t, Time next_t, const VectorXf& next_x,
                                 const std::vector<VectorXf>& us) const {
  float total_cost = 0.0;

  // State costs.
  for (const auto& cost : state_costs_)
    total_cost += cost->Evaluate(next_t, next_x);

  // Control costs.
  for (const auto& pair : control_costs_) {
    const PlayerIndex& player = pair.first;
    const auto& cost = pair.second;

    total_cost += cost->Evaluate(t, us[player]);
  }

  return total_cost;
}

QuadraticCostApproximation PlayerCost::Quadraticize(
    Time t, const VectorXf& x, const std::vector<VectorXf>& us) const {
  QuadraticCostApproximation q(x.size(), state_regularization_);

  // Accumulate state costs.
  for (const auto& cost : state_costs_)
    cost->Quadraticize(t, x, &q.state.hess, &q.state.grad);

  // Accumulate control costs.
  AccumulateControlCosts(control_costs_, t, us, control_regularization_, &q);

  // Accumulate state constraints (including augmented Lagrangian terms scaled
  // by appropriate multipliers).
  for (const auto& constraint : state_constraints_)
    constraint->Quadraticize(t, x, &q.state.hess, &q.state.grad);

  // Accumulate control constraints.
  AccumulateControlConstraints(control_constraints_, t, us,
                               control_regularization_, &q);

  return q;
}

QuadraticCostApproximation PlayerCost::QuadraticizeControlCosts(
    Time t, const VectorXf& x, const std::vector<VectorXf>& us) const {
  QuadraticCostApproximation q(x.size(), state_regularization_);

  // Accumulate control costs.
  AccumulateControlCosts(control_costs_, t, us, control_regularization_, &q);

  return q;
}

}  // namespace ilqgames
