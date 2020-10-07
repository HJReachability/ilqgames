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
// (Time-invariant) affine scalar constraint, i.e., g(x) = a^T x - b.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_CONSTRAINT_AFFINE_SCALAR_CONSTRAINT_H
#define ILQGAMES_CONSTRAINT_AFFINE_SCALAR_CONSTRAINT_H

#include <ilqgames/constraint/time_invariant_constraint.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <memory>
#include <string>

namespace ilqgames {

class AffineScalarConstraint : public TimeInvariantConstraint {
 public:
  ~AffineScalarConstraint() {}
  AffineScalarConstraint(const VectorXf& a, float b, bool is_equality,
                         const std::string& name = "")
      : TimeInvariantConstraint(is_equality, name),
        a_(a),
        b_(b),
        hess_of_sq_(a * a.transpose()) {}

  // Evaluate this constraint value, i.e., g(x).
  float Evaluate(const VectorXf& input) const {
    CHECK_EQ(a_.size(), input.size());
    return a_.transpose() * input - b_;
  }

  // Quadraticize the constraint value and its square, each scaled by lambda or
  // mu, respectively (terms in the augmented Lagrangian).
  void Quadraticize(Time t, const VectorXf& input, MatrixXf* hess,
                    VectorXf* grad) const {
    CHECK_NOTNULL(hess);
    CHECK_NOTNULL(grad);
    CHECK_EQ(input.size(), a_.size());
    CHECK_EQ(hess->rows(), input.size());
    CHECK_EQ(hess->cols(), input.size());
    CHECK_EQ(grad->size(), input.size());

    // Get current lambda and mu.
    const float lambda = Lambda(t);
    const float mu = Mu(t, input);

    // Compute gradient and Hessian.
    (*grad) += lambda * a_ + mu * (hess_of_sq_ * input - b_ * a_);
    (*hess) += mu * hess_of_sq_;
  }

 private:
  // Coefficient vector and nominal value.
  const VectorXf a_;
  const float b_;

  // Precompute Hessian of constraint squared (for speed).
  const MatrixXf hess_of_sq_;
};  //\class AffineScalarConstraint

}  // namespace ilqgames

#endif
