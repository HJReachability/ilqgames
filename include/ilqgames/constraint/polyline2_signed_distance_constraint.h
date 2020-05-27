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
// Constraint on the signed distance to a polyline. Can be oriented either
// `right` or `left`, i.e., can constrain the signed distance to be either > or
// < the given threshold, respectively.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_CONSTRAINT_POLYLINE2_SIGNED_DISTANCE_CONSTRAINT_H
#define ILQGAMES_CONSTRAINT_POLYLINE2_SIGNED_DISTANCE_CONSTRAINT_H

#include <ilqgames/constraint/time_invariant_constraint.h>
#include <ilqgames/cost/semiquadratic_polyline2_cost.h>
#include <ilqgames/geometry/polyline2.h>
#include <ilqgames/utils/types.h>

#include <string>
#include <utility>

namespace ilqgames {

class Polyline2SignedDistanceConstraint : public TimeInvariantConstraint {
 public:
  Polyline2SignedDistanceConstraint(
      const Polyline2& polyline,
      const std::pair<Dimension, Dimension>& position_idxs, float threshold,
      bool oriented_right, const std::string& name = "")
      : TimeInvariantConstraint(name),
        polyline_(polyline),
        signed_threshold_sq_(sgn(threshold) * threshold * threshold),
        oriented_right_(oriented_right),
        xidx_(position_idxs.first),
        yidx_(position_idxs.second) {
    // Set equivalent cost pointer.
    const float new_threshold =
      (oriented_right) ? threshold + kCostBuffer : threshold - kCostBuffer;
    equivalent_cost_.reset(new SemiquadraticPolyline2Cost(
        kEquivalentCostWeight, polyline, position_idxs, new_threshold,
        !oriented_right, name + "/Cost"));
  }

  // Check if this constraint is satisfied, and optionally return the value of a
  // function whose zero sub-level set corresponds to the feasible set.
  bool IsSatisfied(const VectorXf& input, float* level = nullptr) const;

  // Quadraticize this cost at the given time and input, and add to the running
  // sum of gradients and Hessians.
  void Quadraticize(const VectorXf& input, MatrixXf* hess,
                    VectorXf* grad) const;

 private:
  // Polyline to compute distances from.
  const Polyline2 polyline_;

  // Threshold for signed squared distance.
  const float signed_threshold_sq_;

  // Orientation.
  const bool oriented_right_;

  // Position indices.
  const Dimension xidx_, yidx_;
};  //\class Polyline2SignedDistanceConstraint

}  // namespace ilqgames

#endif
