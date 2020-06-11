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
// Orientation cost: agent vs nominal heading.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/cost/orientation_cost.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <math.h>

namespace ilqgames {

float OrientationCost::Evaluate(const VectorXf& input) const {
  CHECK_LT(dim_, input.size());

  // Map heading to x,y coordinates in unit circle.
  const float angle_diff =
      std::fmod(input(dim_) - nominal_ + M_PI, M_PI * 2.0) - M_PI;

  return 0.5 * weight_ * angle_diff * angle_diff;
}

void OrientationCost::Quadraticize(const VectorXf& input, MatrixXf* hess,
                                   VectorXf* grad) const {
  CHECK_LT(dim_, input.size());
  CHECK_NOTNULL(hess);
  CHECK_NOTNULL(grad);

  // Check dimensions.
  CHECK_EQ(input.size(), hess->rows());
  CHECK_EQ(input.size(), hess->cols());
  CHECK_EQ(input.size(), grad->size());

  // Populate Hessian and gradient.
  const float angle_diff =
      std::fmod(input(dim_) - nominal_ + M_PI, M_PI * 2.0) - M_PI;
  float dx = weight_ * angle_diff;
  float ddx = weight_;

  if (IsExponentiated()) {
    const float aw = exponential_constant_ * weight_;
    const float angle_diff_sq = angle_diff * angle_diff;
    const float exp_cost = std::exp(0.5 * aw * angle_diff_sq);

    dx = aw * angle_diff * exp_cost;
    ddx = aw * (aw * angle_diff_sq + 1.0) * exp_cost;
  }

  (*grad)(dim_) += dx;
  (*hess)(dim_, dim_) += ddx;
}

}  // namespace ilqgames
