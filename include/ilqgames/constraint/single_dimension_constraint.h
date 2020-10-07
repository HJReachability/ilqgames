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
// (Time-invariant) single dimension constraint, i.e., g(x) = (+/-) (x_i - d),
// where d is a threshold and sign is determined by the `keep_below` argument
// (positive is true).
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_CONSTRAINT_SINGLE_DIMENSION_CONSTRAINT_H
#define ILQGAMES_CONSTRAINT_SINGLE_DIMENSION_CONSTRAINT_H

#include <ilqgames/constraint/time_invariant_constraint.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <memory>
#include <string>

namespace ilqgames {

class SingleDimensionConstraint : public TimeInvariantConstraint {
 public:
  ~SingleDimensionConstraint() {}
  SingleDimensionConstraint(Dimension dim, float threshold, bool keep_below,
                            const std::string& name = "")
      : TimeInvariantConstraint(false, name),
        dim_(dim),
        threshold_(threshold),
        keep_below_(keep_below) {}

  // Evaluate this constraint value, i.e., g(x).
  float Evaluate(const VectorXf& input) const {
    return (keep_below_) ? input(dim_) - threshold_ : threshold_ - input(dim_);
  }

  // Quadraticize the constraint value and its square, each scaled by lambda or
  // mu, respectively (terms in the augmented Lagrangian).
  void Quadraticize(Time t, const VectorXf& input, MatrixXf* hess,
                    VectorXf* grad) const {
    CHECK_NOTNULL(hess);
    CHECK_NOTNULL(grad);
    CHECK_EQ(hess->rows(), input.size());
    CHECK_EQ(hess->cols(), input.size());
    CHECK_EQ(grad->size(), input.size());

    // Get current lambda.
    const float lambda = Lambda(t);

    // Compute gradient and Hessian.
    const float sign = (keep_below_) ? 1.0 : -1.0;
    const float x = input(dim_);
    const float g = sign * (x - threshold_);

    float dx = sign;
    float ddx = 0.0;
    ModifyDerivatives(t, g, &dx, &ddx);

    (*grad)(dim_) += dx;
    (*hess)(dim_, dim_) += ddx;
  }

 private:
  // Dimension to constrain, threshold value, and sign of constraint.
  const Dimension dim_;
  const float threshold_;
  const bool keep_below_;
};  //\class SingleDimensionConstraint

}  // namespace ilqgames

#endif
