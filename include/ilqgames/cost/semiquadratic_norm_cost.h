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
// Semiquadratic cost function of the norm of two states (difference from some
// nominal norm value), i.e. 0.5 * w * (||(x, y)|| - nominal)^2 if ||(x, y)|| >
// nominal (or optionally <).
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_COST_SEMIQUADRATIC_NORM_COST_H
#define ILQGAMES_COST_SEMIQUADRATIC_NORM_COST_H

#include <ilqgames/cost/time_invariant_cost.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <string>
#include <utility>

namespace ilqgames {

class SemiquadraticNormCost : public TimeInvariantCost {
 public:
  // Construct from a multiplicative weight, the dimensions in which to apply
  // the semiquadratic cost, a threshold, and a flag for which side to apply it.
  SemiquadraticNormCost(float weight,
                        const std::pair<Dimension, Dimension>& dims,
                        float threshold, bool oriented_right,
                        const std::string& name = "")
      : TimeInvariantCost(weight, name),
        dim1_(dims.first),
        dim2_(dims.second),
        threshold_(threshold),
        oriented_right_(oriented_right) {
    CHECK_GE(dim1_, 0);
    CHECK_GE(dim2_, 0);
  }

  // Evaluate this cost at the current input.
  float Evaluate(const VectorXf& input) const;

  // Quadraticize this cost at the given input, and add to the running
  // sum of gradients and Hessians.
  void Quadraticize(const VectorXf& input, MatrixXf* hess,
                    VectorXf* grad) const;

 private:
  // Dimensions in which to apply the quadratic cost.
  const Dimension dim1_, dim2_;

  // Threshold and which side to apply it to.
  const float threshold_;
  const bool oriented_right_;
};  //\class SemiquadraticNormCost

}  // namespace ilqgames

#endif
