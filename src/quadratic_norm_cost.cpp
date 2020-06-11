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
// Quadratic cost function of the norm of two states (difference from some
// nominal norm value), i.e. 0.5 * w * (||(x, y)|| - nominal)^2.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/cost/quadratic_norm_cost.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>

namespace ilqgames {

float QuadraticNormCost::Evaluate(const VectorXf& input) const {
  CHECK_LT(dim1_, input.size());
  CHECK_LT(dim2_, input.size());

  // Otherwise, cost is squared 2-norm of entire input.
  const float diff = std::hypot(input(dim1_), input(dim2_)) - nominal_;
  return 0.5 * weight_ * diff * diff;
}

void QuadraticNormCost::Quadraticize(const VectorXf& input, MatrixXf* hess,
                                     VectorXf* grad) const {
  CHECK_LT(dim1_, input.size());
  CHECK_LT(dim2_, input.size());
  CHECK_NOTNULL(hess);
  CHECK_NOTNULL(grad);

  // Check dimensions.
  CHECK_EQ(input.size(), hess->rows());
  CHECK_EQ(input.size(), hess->cols());
  CHECK_EQ(input.size(), grad->size());

  // Populate Hessian and gradient.
  const float x = input(dim1_);
  const float y = input(dim2_);
  const float x_sq = x * x;
  const float y_sq = y * y;
  const float norm_sq = x_sq + y_sq;
  const float norm = std::sqrt(norm_sq);
  const float norm_3 = norm * norm_sq;

  float dx = -weight_ * x * (-1.0 + nominal_ / norm);
  float dy = -weight_ * y * (-1.0 + nominal_ / norm);
  float ddx = weight_ - (nominal_ * y_sq * weight_) / norm_3;
  float ddy = weight_ - (nominal_ * x_sq * weight_) / norm_3;
  float dxdy = nominal_ * x * y * weight_ / norm_3;

  if (IsExponentiated()) {
    const float aw = exponential_constant_ * weight_;
    const float diff = norm - nominal_;
    const float aw_diff_sq = aw * diff * diff;
    const float exp_cost = std::exp(0.5 * aw_diff_sq);

    dx = aw * x * diff * exp_cost / norm;
    dy = aw * y * diff * exp_cost / norm;

    ddx =
        -aw *
        (x_sq * (diff * norm - norm_sq * (aw_diff_sq + 1.0)) - diff * norm_3) *
        exp_cost / (norm_sq * norm_sq);
    ddy =
        -aw *
        (y_sq * (diff * norm - norm_sq * (aw_diff_sq + 1.0)) - diff * norm_3) *
        exp_cost / (norm_sq * norm_sq);
    dxdy = -aw * x * y * (diff - norm * (aw_diff_sq + 1.0)) * exp_cost / norm_3;
  }

  (*hess)(dim1_, dim1_) += ddx;
  (*hess)(dim2_, dim2_) += ddy;
  (*hess)(dim1_, dim2_) += dxdy;
  (*hess)(dim2_, dim1_) += dxdy;

  (*grad)(dim1_) += dx;
  (*grad)(dim2_) += dy;
}

}  // namespace ilqgames
