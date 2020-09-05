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
// Signed distance from a given polyline.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_COST_POLYLINE2_SIGNED_DISTANCE_COST_H
#define ILQGAMES_COST_POLYLINE2_SIGNED_DISTANCE_COST_H

#include <ilqgames/cost/time_invariant_cost.h>
#include <ilqgames/geometry/polyline2.h>
#include <ilqgames/utils/types.h>

#include <string>
#include <tuple>

namespace ilqgames {

class Polyline2SignedDistanceCost : public TimeInvariantCost {
 public:
  // Construct from a multiplicative weight and the input dimensions
  // corresponding to (x, y)-position.
  Polyline2SignedDistanceCost(
      const Polyline2& polyline,
      const std::pair<Dimension, Dimension>& position_idxs,
      const float nominal = 0.0, bool oriented_same_as_polyline = true,
      const std::string& name = "")
      : TimeInvariantCost(1.0, name),
        polyline_(polyline),
        xidx_(position_idxs.first),
        yidx_(position_idxs.second),
        nominal_(nominal),
        oriented_same_as_polyline_(oriented_same_as_polyline) {}

  // Evaluate this cost at the current input.
  float Evaluate(const VectorXf& input) const;

  // Quadraticize this cost at the given input, and add to the running
  // sum of gradients and Hessians.
  void Quadraticize(const VectorXf& input, MatrixXf* hess,
                    VectorXf* grad) const;

 private:
  // Polyline to compute distances from.
  const Polyline2 polyline_;

  // Dimensions of input corresponding to (x, y)-position.
  const Dimension xidx_;
  const Dimension yidx_;

  // Nominal value.
  const float nominal_;

  // Whether the orientation is the same or opposite that of the polyline.
  const bool oriented_same_as_polyline_;
};  //\class Polyline2SignedDistanceCost

}  // namespace ilqgames

#endif
