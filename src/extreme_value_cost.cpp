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

#include <ilqgames/cost/extreme_value_cost.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <numeric>

namespace ilqgames {

float ExtremeValueCost::Evaluate(Time t, const VectorXf& input) const {
  float evaluated;
  ExtremeCost(t, input, &evaluated);

  return evaluated;
}

void ExtremeValueCost::Quadraticize(Time t, const VectorXf& input,
                                    MatrixXf* hess, VectorXf* grad) const {
  const Cost* extreme_cost = ExtremeCost(t, input);

  // Call that cost's 'Quadraticize' function.
  extreme_cost->Quadraticize(t, input, hess, grad);
}

const Cost* ExtremeValueCost::ExtremeCost(Time t, const VectorXf& input,
                                          float* evaluated) const {
  const Cost* extreme_cost;
  float extreme_value = (is_min_) ? std::numeric_limits<float>::infinity()
                                  : -std::numeric_limits<float>::infinity();

  for (const auto& cost : costs_) {
    const float value = cost->Evaluate(t, input);
    if (!(!(is_min_ && value < extreme_value) &&
          !(!is_min_ && value > extreme_value))) {
      extreme_value = value;
      extreme_cost = cost.get();
      if (evaluated) *evaluated = value;
    }
  }

  CHECK_NOTNULL(extreme_cost);

  return extreme_cost;
}

}  // namespace ilqgames
