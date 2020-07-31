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
// Cost which represents the min (or optionally, max) of a set of other costs.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_COST_EXTREME_VALUE_COST_H
#define ILQGAMES_COST_EXTREME_VALUE_COST_H

#include <ilqgames/cost/time_invariant_cost.h>
#include <ilqgames/utils/types.h>

#include <memory>
#include <string>

namespace ilqgames {

class ExtremeValueCost : public Cost {
 public:
  ExtremeValueCost(const std::vector<std::shared_ptr<const Cost>>& costs,
                   bool is_min, const std::string& name = "")
      : Cost(1.0, name), is_min_(is_min), costs_(costs) {}

  // Evaluate this cost at the current time and input.
  float Evaluate(Time t, const VectorXf& input) const;

  // Quadraticize this cost at the given time and input, and add to the running
  // sum of gradients and Hessians.
  void Quadraticize(Time t, const VectorXf& input, MatrixXf* hess,
                    VectorXf* grad) const;

  // Return a pointer to the extreme cost, and optionally evaluate it too.
  const Cost* ExtremeCost(Time t, const VectorXf& input,
                          float* evaluated = nullptr) const;

 private:
  // Is this a min or a max?
  const bool is_min_;

  // List of sub-costs.
  const std::vector<std::shared_ptr<const Cost>> costs_;
};  //\class ExtremeValueCost

}  // namespace ilqgames

#endif
