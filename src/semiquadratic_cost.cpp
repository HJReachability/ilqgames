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
// Quadratic cost in a particular (or all) dimension(s).
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/cost/semiquadratic_cost.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>

namespace ilqgames {

// Evaluate this cost at the current input.
float SemiquadraticCost::Evaluate(const VectorXf& input) const {
  CHECK_LT(dimension_, input.size());

  const float diff = input(dimension_) - threshold_;
  if ((diff > 0.0 && oriented_right_) || (diff < 0.0 && !oriented_right_))
    return 0.5 * weight_ * diff * diff;

  return 0.0;
}

// Quadraticize this cost at the given input, and add to the running
// sum of gradients and Hessians (if non-null).
void SemiquadraticCost::Quadraticize(const VectorXf& input, MatrixXf* hess,
                                     VectorXf* grad) const {
  CHECK_LT(dimension_, input.size());

  // Handle no cost case first.
  const float diff = input(dimension_) - threshold_;
  if ((diff < 0.0 && oriented_right_) || (diff > 0.0 && !oriented_right_))
    return;

  // Check dimensions.
  CHECK_NOTNULL(hess);
  CHECK_NOTNULL(grad);
  CHECK_EQ(input.size(), hess->rows());
  CHECK_EQ(input.size(), hess->cols());
  CHECK_EQ(input.size(), grad->size());

  // Compute gradient and Hessian.
  float dx = weight_ * diff;
  float ddx = weight_;

  ModifyDerivatives(input, &dx, &ddx);

  (*grad)(dimension_) += dx;
  (*hess)(dimension_, dimension_) += ddx;
}

}  // namespace ilqgames
