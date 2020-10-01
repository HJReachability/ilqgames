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
// Base class for all explicit (scalar-valued) equality constraints. These
// constraints are of the form: g(x) = 0 for some vector x.
//
// In addition to checking for satisfaction (and returning the squared norm of
// the constraint value g(x)), they also support computing first and second
// derivatives of the constraint value itself and the square of the constraint
// value, each scaled by lambda or mu respectively (from the augmented
// Lagrangian).
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_CONSTRAINT_EQUALITY_CONSTRAINT_H
#define ILQGAMES_CONSTRAINT_EQUALITY_CONSTRAINT_H

#include <ilqgames/utils/relative_time_tracker.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <memory>
#include <string>

namespace ilqgames {

class EqualityConstraint : public RelativeTimeTracker {
 public:
  virtual ~EqualityConstraint() {}

  // Check if this constraint is satisfied, and optionally return the constraint
  // value, which equals zero if the constraint is satisfied.
  virtual bool IsSatisfied(Time t, const VectorXf& input,
                           float* level) const = 0;

  // Quadraticize the constraint value and its square, each scaled by lambda or
  // mu, respectively (terms in the augmented Lagrangian).
  virtual void Quadraticize(Time t, size_t time_step, const VectorXf& input,
                            MatrixXf* hess, VectorXf* grad) const = 0;

  // Accessors and setters.
  float& Lambda(size_t time_step) { return lambdas_[time_step]; }
  static float& Mu() { return mu_; }

 protected:
  explicit EqualityConstraint(size_t num_time_steps, const std::string& name)
      : RelativeTimeTracker(name), lambdas_(num_time_steps, 0.0) {}

  // Name of this constraint.
  const std::string name_;

  // Multipliers, one per time step. Also a static augmented multiplier for an
  // augmented Lagrangian.
  std::vector<float> lambdas_;
  static float mu_;
};  //\class EqualityConstraint

}  // namespace ilqgames

#endif
