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
// nominal norm value), i.e. 0.5 * w * (||(x, y)|| - nominal)^2 ||(x, y)|| >
// nominal (or optionally <).
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/cost/semiquadratic_norm_cost.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>

namespace ilqgames {

float SemiquadraticNormCost::Evaluate(const VectorXf& input) const {
  CHECK_LT(dim1_, input.size());
  CHECK_LT(dim2_, input.size());

  const float diff = std::hypot(input(dim1_), input(dim2_)) - threshold_;
  if ((diff > 0.0 && oriented_right_) || (diff < 0.0 && !oriented_right_))
    return 0.5 * weight_ * diff * diff;

  return 0.0;
}

void SemiquadraticNormCost::Quadraticize(const VectorXf& input, MatrixXf* hess,
                                         VectorXf* grad) const {
  CHECK_LT(dim1_, input.size());
  CHECK_LT(dim2_, input.size());
  CHECK_NOTNULL(hess);
  CHECK_NOTNULL(grad);

  // Check dimensions.
  CHECK_EQ(input.size(), hess->rows());
  CHECK_EQ(input.size(), hess->cols());
  CHECK_EQ(input.size(), grad->size());

  // Check if cost is active.
  const float norm = std::hypot(input(dim1_), input(dim2_));
  if ((norm > threshold_ && !oriented_right_) ||
      (norm < threshold_ && oriented_right_))
    return;

  // Populate hessian and, optionally, gradient.
  const float norm_2 = norm * norm;
  const float norm_3 = norm * norm_2;
  const float x = input(dim1_);
  const float y = input(dim2_);
  const float dx = -weight_ * x * (-1.0 + threshold_ / norm);
  const float dy = -weight_ * y * (-1.0 + threshold_ / norm);
  const float ddx = weight_ - (threshold_ * y * y * weight_) / norm_3;
  const float ddy = weight_ - (threshold_ * x * x * weight_) / norm_3;
  const float dxdy = threshold_ * x * y * weight_ / norm_3;

  (*grad)(dim1_) += dx;
  (*grad)(dim2_) += dy;

  (*hess)(dim1_, dim1_) += ddx;
  (*hess)(dim2_, dim2_) += ddy;
  (*hess)(dim1_, dim2_) += dxdy;
  (*hess)(dim2_, dim1_) += dxdy;
}

}  // namespace ilqgames
