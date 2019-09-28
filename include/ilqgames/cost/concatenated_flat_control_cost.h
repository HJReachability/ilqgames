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
// Specialization of generalized control costs for flat systems.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_COST_CONCATENATED_FLAT_CONTROL_COST_H
#define ILQGAMES_COST_CONCATENATED_FLAT_CONTROL_COST_H

#include <ilqgames/cost/flat_control_cost.h>
#include <ilqgames/dynamics/concatenated_flat_system.h>
#include <ilqgames/utils/types.h>

#include <string>

namespace ilqgames {

class ConcatenatedFlatControlCost : public FlatControlCost {
 public:
  virtual ~ConcatenatedFlatControlCost() {}

  // Evaluate this cost at the current time and inputs.
  virtual float Evaluate(const VectorXf& xi, const VectorXf& v) const = 0;

  // Quadraticize this cost at the given time and inputs, and add to the running
  // sum of state gradients and state/control Hessians (if non-null).
  virtual void Quadraticize(const VectorXf& xi, const VectorXf& v,
                            MatrixXf* hess_v, MatrixXf* hess_xi,
                            VectorXf* grad_xi) const = 0;

 protected:
  // Accepts weight and name like usual, but also pointer to flat dynamics.
  // Also records which subsystem's control the cost will correspond to.
  ConcatenatedFlatControlCost(
      float weight, const std::shared_ptr<const ConcatenatedFlatSystem>& dynamics,
      PlayerIndex subsystem_idx, const std::string& name = "")
      : FlatControlCost(weight, dynamics, name),
        subsystem_idx_(subsystem_idx) {}

  // Index of subsystem whose input this cost corresponds to.
  const PlayerIndex subsystem_idx_;
};  //\class ConcatenatedFlatControlCost

}  // namespace ilqgames

#endif
