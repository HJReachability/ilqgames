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
// Constraint on the value of a single dimension of the input. This constraint
// can be oriented either `left` or `right`, i.e., enforcing that the input is <
// or > the specified threshold, respectively.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/constraint/single_dimension_constraint.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <string>
#include <utility>

namespace ilqgames {

bool SingleDimensionConstraint::IsSatisfied(const VectorXf& input,
                                            float* level) const {
  // Sign corresponding to the orientation of this constraint.
  const float sign = (oriented_right_) ? 1.0 : -1.0;

  // Maybe populate level.
  const float delta = threshold_ - input(dimension_);
  if (level) *level = sign * delta;

  return (oriented_right_) ? delta < 0.0 : delta > 0.0;
}

void SingleDimensionConstraint::Quadraticize(const VectorXf& input,
                                             MatrixXf* hess,
                                             VectorXf* grad) const {
  CHECK_NOTNULL(hess);
  CHECK_NOTNULL(grad);

  // Check dimensions.
  CHECK_EQ(input.size(), hess->rows());
  CHECK_EQ(input.size(), hess->cols());
  CHECK_EQ(input.size(), grad->size());

  // Compute Hessian and gradient.
  const float delta_inv = 1.0 / (threshold_ - input(dimension_));
  const float weighted_delta_inv = weight_ * delta_inv;
  (*grad)(dimension_) += weighted_delta_inv;
  (*hess)(dimension_, dimension_) += weighted_delta_inv * delta_inv;
}

}  // namespace ilqgames
