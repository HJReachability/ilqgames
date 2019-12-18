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

  // Populate hessian and, optionally, gradient.
  const float norm = std::hypot(input(dim1_), input(dim2_));
  const float norm_3 = norm * norm * norm;
  (*hess)(dim1_, dim1_) +=
      weight_ - (nominal_ * input(dim2_) * input(dim2_) * weight_) / norm_3;
  (*hess)(dim2_, dim2_) +=
      weight_ - (nominal_ * input(dim1_) * input(dim1_) * weight_) / norm_3;
  (*hess)(dim1_, dim2_) +=
      nominal_ * input(dim1_) * input(dim2_) * weight_ / norm_3;
  (*hess)(dim2_, dim1_) += (*hess)(dim1_, dim2_);

  (*grad)(dim1_) += -weight_ * input(dim1_) * (-1.0 + nominal_ / norm);
  (*grad)(dim2_) += -weight_ * input(dim2_) * (-1.0 + nominal_ / norm);
}

}  // namespace ilqgames
