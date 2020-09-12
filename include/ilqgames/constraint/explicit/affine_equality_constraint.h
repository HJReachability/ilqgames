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
// (Time-invariant) linear equality constraint, i.e., a^T x - b = 0.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_CONSTRAINT_EXPLICIT_AFFINE_EQUALITY_CONSTRAINT_H
#define ILQGAMES_CONSTRAINT_EXPLICIT_AFFINE_EQUALITY_CONSTRAINT_H

#include <ilqgames/constraint/explicit/time_invariant_equality_constraint.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <memory>
#include <string>

namespace ilqgames {

class AffineEqualityConstraint : public TimeInvariantEqualityConstraint {
 public:
  ~AffineEqualityConstraint() {}
  AffineEqualityConstraint(const VectorXf& a, float b,
                           const std::string& name = "")
      : TimeInvariantEqualityConstraint(name), a_(a), b_(b) {}

  // Check if this constraint is satisfied, and optionally return the constraint
  // value, which equals zero if the constraint is satisfied.
  bool IsSatisfied(const VectorXf& input, float* level) const {
    CHECK_EQ(input.size(), a_.size());
    const float value = a_.transpose() * input + b_;

    if (*level) *level = value;
    return std::abs(value) < constants::kSmallNumber;
  }

  // Quadraticize the constraint value. Do *not* keep a running sum since we
  // keep separate multipliers for each constraint.
  void Quadraticize(const VectorXf& input, MatrixXf* hess,
                    VectorXf* grad) const {
    CHECK_NOTNULL(hess);
    hess->resize(input.size(), input.size());

    CHECK_NOTNULL(grad);
    grad->resize(input.size());

    hess->setZero();
    (*grad) = a_;
  }

 private:
  // Coefficient vector and nominal value.
  const VectorXf a_;
  const float b_;
};  //\class AffineEqualityConstraint

}  // namespace ilqgames

#endif
