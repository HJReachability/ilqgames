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
// (Time-invariant) vector equality constraint, i.e., ||x - \hat x|| = 0.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_CONSTRAINT_VECTOR_EQUALITY_CONSTRAINT_H
#define ILQGAMES_CONSTRAINT_VECTOR_EQUALITY_CONSTRAINT_H

#include <ilqgames/constraint/time_invariant_equality_constraint.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <memory>
#include <string>

namespace ilqgames {

class VectorEqualityConstraint : public TimeInvariantEqualityConstraint {
 public:
  ~VectorEqualityConstraint() {}
  VectorEqualityConstraint(const VectorXf& nominal,
                           const std::string& name = "")
      : TimeInvariantEqualityConstraint(name), nominal_(nominal) {}

  // Check if this constraint is satisfied, and optionally return the constraint
  // value, which equals zero if the constraint is satisfied.
  bool IsSatisfied(const VectorXf& input, float* level) const {
    CHECK_EQ(input.size(), nominal_.size());
    const float value = (input - nominal_).norm();

    if (*level) *level = value;
    return std::abs(value) < constants::kSmallNumber;
  }

  // Quadraticize the constraint value and its square, each scaled by lambda or
  // mu, respectively (terms in the augmented Lagrangian).
  void Quadraticize(float lambda, float mu, const VectorXf& input,
                    MatrixXf* hess, VectorXf* grad) const {
    CHECK_NOTNULL(hess);
    CHECK_NOTNULL(grad);
    CHECK_EQ(input.size(), nominal_.size());
    CHECK_EQ(hess->rows(), input.size());
    CHECK_EQ(hess->cols(), input.size());
    CHECK_EQ(grad->size(), input.size());

    // Compute value of the constraint.
    const VectorXf delta = input - nominal_;
    const float value = delta.norm();

    // Compute gradient and Hessian.
    (*grad) += (mu + lambda / value) * delta;
    (*hess) -= (lambda / (value * value * value)) * delta * delta.transpose();
    hess->diagonal() += VectorXf::Constant(input.size(), mu + lambda / value);
  }

 private:
  // Nominal vector.
  const VectorXf nominal_;
};  //\class VectorEqualityConstraint

}  // namespace ilqgames

#endif
