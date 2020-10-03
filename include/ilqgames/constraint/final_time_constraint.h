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
// Equality constraint that is only active after the threshold time.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_CONSTRAINT_FINAL_TIME_CONSTRAINT_H
#define ILQGAMES_CONSTRAINT_FINAL_TIME_CONSTRAINT_H

#include <ilqgames/constraint/constraint.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <memory>
#include <string>

namespace ilqgames {

class FinalTimeConstraint : public Constraint {
 public:
  ~FinalTimeConstraint() {}
  FinalTimeConstraint(const std::shared_ptr<Constraint>& constraint,
                      Time threshold_time)
      : Constraint(*constraint),
        constraint_(constraint),
        threshold_time_(threshold_time) {
    CHECK_NOTNULL(constraint_);
  }

  // Evaluate this constraint value, i.e., g(x).
  float Evaluate(Time t, const VectorXf& input) const {
    return (t < initial_time_ + threshold_time_)
               ? 0.0
               : constraint_->Evaluate(t, input);
  }

  // Quadraticize the constraint value and its square, each scaled by lambda or
  // mu, respectively (terms in the augmented Lagrangian).
  void Quadraticize(Time t, const VectorXf& input, MatrixXf* hess,
                    VectorXf* grad) const {
    if (t >= initial_time_ + threshold_time_)
      constraint_->Quadraticize(t, input, hess, grad);
  }

 private:
  // Underlying constraint.
  const std::shared_ptr<Constraint> constraint_;

  // Time threshold relative to initial time after which to apply constraint.
  const Time threshold_time_;
};  //\class FinalTimeConstraint

}  // namespace ilqgames

#endif
