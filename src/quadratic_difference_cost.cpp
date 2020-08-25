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
// Quadratic cost on pairwise differences between dimensions.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/cost/quadratic_difference_cost.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>

namespace ilqgames {

float QuadraticDifferenceCost::Evaluate(const VectorXf& input) const {
  float total = 0.0;
  for (size_t ii = 0; ii < dims1_.size(); ii++) {
    const float diff = input(dims1_[ii]) - input(dims2_[ii]);
    total += diff * diff;
  }

  // Otherwise, cost is squared 2-norm of entire input.
  return 0.5 * weight_ * total;
}

void QuadraticDifferenceCost::Quadraticize(const VectorXf& input,
                                           MatrixXf* hess,
                                           VectorXf* grad) const {
  CHECK_NOTNULL(hess);

  // Check dimensions.
  CHECK_EQ(input.size(), hess->rows());
  CHECK_EQ(input.size(), hess->cols());

  if (grad) CHECK_EQ(input.size(), grad->size());

  for (size_t ii = 0; ii < dims1_.size(); ii++) {
    const float dx = weight_ * (input(dims1_[ii]) - input(dims2_[ii]));
    const float dy = -dx;
    const float ddx = weight_;
    const float ddy = weight_;
    const float dxdy = -weight_;

    (*hess)(dims1_[ii], dims1_[ii]) += ddx;
    (*hess)(dims2_[ii], dims2_[ii]) += ddy;
    (*hess)(dims1_[ii], dims2_[ii]) += dxdy;
    (*hess)(dims2_[ii], dims1_[ii]) += dxdy;

    if (grad) {
      (*grad)(dims1_[ii]) += dx;
      (*grad)(dims2_[ii]) += dy;
    }
  }
}

}  // namespace ilqgames
