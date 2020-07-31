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
// Nominal value minus distance between two points in the given dimensions.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_COST_SIGNED_DISTANCE_COST_H
#define ILQGAMES_COST_SIGNED_DISTANCE_COST_H

#include <ilqgames/cost/time_invariant_cost.h>
#include <ilqgames/utils/types.h>

#include <string>

namespace ilqgames {

class SignedDistanceCost : public TimeInvariantCost {
 public:
  SignedDistanceCost(const std::pair<Dimension, Dimension>& dims1,
                     const std::pair<Dimension, Dimension>& dims2,
                     float nominal = 0.0, bool less_is_positive = true,
                     const std::string& name = "")
      : TimeInvariantCost(1.0, name),
        xdim1_(dims1.first),
        ydim1_(dims1.second),
        xdim2_(dims2.first),
        ydim2_(dims2.second),
        nominal_(nominal),
        less_is_positive_(less_is_positive) {
    CHECK_GE(xdim1_, 0);
    CHECK_GE(ydim1_, 0);
    CHECK_GE(xdim2_, 0);
    CHECK_GE(ydim2_, 0);
  }

  // Evaluate this cost at the current input.
  float Evaluate(const VectorXf& input) const;

  // Distanceize this cost at the given input, and add to the running
  // sum of gradients and Hessians.
  void Quadraticize(const VectorXf& input, MatrixXf* hess,
                    VectorXf* grad) const;

 private:
  // Dimensions in which to apply the distance cost.
  const Dimension xdim1_, ydim1_;
  const Dimension xdim2_, ydim2_;

  // Nominal value in this (or all) dimensions.
  const float nominal_;

  // Orientation. Whether less than nominal is positive or negative.
  const bool less_is_positive_;
};  //\class SignedDistanceCost

}  // namespace ilqgames

#endif
