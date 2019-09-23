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
// Quadratic generalized control cost for the flat 4D unicycle dynamics.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_COST_QUADRATIC_CONTROL_COST_FLAT_UNICYCLE_4D_H
#define ILQGAMES_COST_QUADRATIC_CONTROL_COST_FLAT_UNICYCLE_4D_H

#include <ilqgames/cost/concatenated_flat_control_cost.h>
#include <ilqgames/dynamics/concatenated_flat_system.h>
#include <ilqgames/utils/types.h>

#include <string>

namespace ilqgames {

class QuadraticControlCostFlatUnicycle4D : public ConcatenatedFlatControlCost {
 public:
  // Construct from a multiplicative weight and the dimension in which to apply
  // the quadratic cost (difference from nominal). If dimension < 0, then
  // applies to all dimensions (i.e. ||input - nominal * ones()||^2).
  QuadraticControlCostFlatUnicycle4D(
      float weight,
      const std::shared_ptr<const ConcatenatedFlatSystem>& dynamics,
      PlayerIndex subsystem_idx, Dimension dim = -1, float nominal = 0.0,
      const std::string& name = "")
      : ConcatenatedFlatControlCost(weight, dynamics, subsystem_idx, name),
        dimension_(dim),
        nominal_(nominal) {}

  // Evaluate this cost at the current inputs.
  float Evaluate(const VectorXf& xi, const VectorXf& v) const;

  // Quadraticize this cost at the given inputs, and add to the running
  // sum of state gradients and state/control Hessians (if non-null).
  void Quadraticize(const VectorXf& xi, const VectorXf& v, MatrixXf* hess_v,
                    MatrixXf* hess_xi, VectorXf* grad_xi) const;

 private:
  // Dimension in which to apply the quadratic cost.
  const Dimension dimension_;

  // Nominal value in this (or all) dimensions.
  const float nominal_;
};  //\class QuadraticControlCostFlatUnicycle4D

}  // namespace ilqgames

#endif
