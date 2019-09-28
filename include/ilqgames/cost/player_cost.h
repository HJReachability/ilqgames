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

#include <ilqgames/cost/cost.h>
#include <ilqgames/cost/generalized_control_cost.h>
#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/quadratic_cost_approximation.h>
#include <ilqgames/utils/types.h>

#include <unordered_map>

namespace ilqgames {

class PlayerCost {
 public:
  // Add new state and control costs for this player.
  void AddStateCost(const std::shared_ptr<Cost>& cost);
  void AddControlCost(PlayerIndex idx, const std::shared_ptr<Cost>& cost);
  void AddGeneralizedControlCost(
      PlayerIndex idx, const std::shared_ptr<GeneralizedControlCost>& cost);

  // Evaluate this cost at the current time, state, and controls, or integrate
  // over an entire trajectory.
  float Evaluate(Time t, const VectorXf& x,
                 const std::vector<VectorXf>& us) const;
  float Evaluate(const OperatingPoint& op, Time time_step) const;

  // Quadraticize this cost at the given time, state, and controls.
  QuadraticCostApproximation Quadraticize(
      Time t, const VectorXf& x, const std::vector<VectorXf>& us) const;

  // Accessors.
  const std::vector<std::shared_ptr<Cost>> StateCosts() const {
    return state_costs_;
  }
  const CostMap<Cost>& ControlCosts() const { return control_costs_; }
  const CostMap<GeneralizedControlCost>& GeneralizedControlCosts() const {
    return generalized_control_costs_;
  }

 private:
  // State costs, control costs, and generalized control costs.
  std::vector<std::shared_ptr<Cost>> state_costs_;
  CostMap<Cost> control_costs_;
  CostMap<GeneralizedControlCost> generalized_control_costs_;
};  //\class PlayerCost

}  // namespace ilqgames

#endif
