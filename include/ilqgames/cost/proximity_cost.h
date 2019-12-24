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
// Penalizes (thresh - relative distance)^2 between two pairs of state
// dimensions (representing two positions of vehicles whose states have been
// concatenated) whenever relative distance is less than thresh.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_COST_PROXIMITY_COST_H
#define ILQGAMES_COST_PROXIMITY_COST_H

#include <ilqgames/cost/time_invariant_cost.h>
#include <ilqgames/utils/types.h>

#include <string>
#include <utility>

namespace ilqgames {

class ProximityCost : public TimeInvariantCost {
 public:
  ProximityCost(float weight,
                const std::pair<Dimension, Dimension>& position_idxs1,
                const std::pair<Dimension, Dimension>& position_idxs2,
                float threshold, const std::string& name = "")
      : TimeInvariantCost(weight, name),
        threshold_(threshold),
        threshold_sq_(threshold * threshold),
        xidx1_(position_idxs1.first),
        yidx1_(position_idxs1.second),
        xidx2_(position_idxs2.first),
        yidx2_(position_idxs2.second) {}

  // Evaluate this cost at the current input.
  float Evaluate(const VectorXf& input) const;

  // Quadraticize this cost at the given input, and add to the running
  // sum of gradients and Hessians.
  void Quadraticize(const VectorXf& input, MatrixXf* hess,
                    VectorXf* grad) const;

 private:
  // Threshold for minimum squared relative distance.
  const float threshold_, threshold_sq_;

  // Position indices for two vehicles.
  const Dimension xidx1_, yidx1_;
  const Dimension xidx2_, yidx2_;
};  //\class ProximityCost

}  // namespace ilqgames

#endif
