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
// Constraint on proximity between two pairs of state dimensions (representing
// 2D position of vehicles whose states have been concatenated). Can be oriented
// either `inside` or `outside`, i.e., can constrain the states to be close
// together or far apart (respectively).
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_CONSTRAINT_PROXIMITY_CONSTRAINT_H
#define ILQGAMES_CONSTRAINT_PROXIMITY_CONSTRAINT_H

#include <ilqgames/constraint/time_invariant_constraint.h>
#include <ilqgames/cost/proximity_cost.h>
#include <ilqgames/utils/types.h>

#include <string>
#include <utility>
#include <math.h>

namespace ilqgames {

class ProximityConstraint : public TimeInvariantConstraint {
 public:
  ProximityConstraint(const std::pair<Dimension, Dimension>& position_idxs1,
                      const std::pair<Dimension, Dimension>& position_idxs2,
                      float threshold, bool inside = false,
                      const std::string& name = "")
      : TimeInvariantConstraint(name),
        threshold_sq_(threshold * threshold),
        inside_(inside),
        xidx1_(position_idxs1.first),
        yidx1_(position_idxs1.second),
        xidx2_(position_idxs2.first),
        yidx2_(position_idxs2.second) {
    // Set equivalent cost pointer.
    const float new_threshold = std::max<float>(threshold - kCostBuffer, 0.0);
    equivalent_cost_.reset(new ProximityCost(kEquivalentCostWeight,
                                             position_idxs1, position_idxs2,
                                             new_threshold, name + "/Cost"));
  }

  // Check if this constraint is satisfied, and optionally return the value of a
  // function whose zero sub-level set corresponds to the feasible set.
  bool IsSatisfiedLevel(const VectorXf& input, float* level) const;

  // Quadraticize this cost at the given time and input, and add to the running
  // sum of gradients and Hessians.
  void Quadraticize(const VectorXf& input, MatrixXf* hess,
                    VectorXf* grad) const;

 private:
  // Threshold for squared relative distance.
  const float threshold_sq_;

  // Orientation, either `inside` (states should be close) or `outside` (states
  // should be far apart).
  const bool inside_;

  // Position indices for two vehicles.
  const Dimension xidx1_, yidx1_;
  const Dimension xidx2_, yidx2_;
};  //\class ProximityConstraint

}  // namespace ilqgames

#endif
