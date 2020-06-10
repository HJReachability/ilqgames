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

#include <ilqgames/cost/quadratic_cost.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <numeric>

namespace ilqgames {

float QuadraticCost::Evaluate(const VectorXf& input) const {
  CHECK_LT(dimension_, input.size());

  // If dimension non-negative, then just square the desired dimension.
  if (dimension_ >= 0) {
    const float delta = input(dimension_) - nominal_;
    return 0.5 * weight_ * delta * delta;
  }

  // Otherwise, cost is squared 2-norm of entire input.
  return 0.5 * weight_ *
         (input - VectorXf::Constant(input.size(), nominal_)).squaredNorm();
}

void QuadraticCost::Quadraticize(const VectorXf& input, MatrixXf* hess,
                                 VectorXf* grad) const {
  CHECK_LT(dimension_, input.size());
  CHECK_NOTNULL(hess);
  CHECK_NOTNULL(grad);

  // Check dimensions.
  CHECK_EQ(input.size(), hess->rows());
  CHECK_EQ(input.size(), hess->cols());
  CHECK_EQ(input.size(), grad->size());

  // Handle single dimension case first.
  const float aw = exponential_constant_ * weight_;

  if (dimension_ >= 0) {
    const float delta = input(dimension_) - nominal_;
    float dx = weight_ * delta;
    float ddx = weight_;

    if (IsExponentiated()) {
      const float aw_delta = aw * delta;
      const float exp_cost = std::exp(0.5 * aw_delta * delta);

      dx = aw_delta * exp_cost;
      ddx = aw * (aw * dx * dx + 1.0) * exp_cost;
    }

    (*grad)(dimension_) += dx;
    (*hess)(dimension_, dimension_) += ddx;
  }

  // Handle dimension < 0 case.
  else {
    const VectorXf delta = input - VectorXf::Constant(input.size(), nominal_);

    if (IsExponentiated()) {
      const float exp_cost = std::exp(0.5 * aw * delta.squaredNorm());
      const float aw_sq = aw * aw;
      VectorXf delta_sq(delta.size());

      *grad += aw * delta * exp_cost;

      for (size_t ii = 0; ii < hess->rows(); ii++) {
        delta_sq(ii) = delta(ii) * delta(ii);

        for (size_t jj = 0; jj < hess->cols(); jj++) {
          if (ii == jj) continue;
          (*hess)(ii, jj) += aw_sq * delta(ii) * delta(jj) * exp_cost;
        }
      }

      hess->diagonal() =
          hess->diagonal() +
          exp_cost * aw *
              (aw * delta_sq + VectorXf::Constant(delta.size(), 1.0));
    } else {
      *grad += weight_ * delta;
      hess->diagonal() =
          hess->diagonal() + VectorXf::Constant(input.size(), weight_);
    }
  }
}

}  // namespace ilqgames
