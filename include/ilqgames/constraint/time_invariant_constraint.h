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
// Base class for all time-invariant constraints.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_CONSTRAINT_TIME_INVARIANT_CONSTRAINT_H
#define ILQGAMES_CONSTRAINT_TIME_INVARIANT_CONSTRAINT_H

#include <ilqgames/constraint/constraint.h>
#include <ilqgames/cost/cost.h>
#include <ilqgames/utils/types.h>

#include <string>

namespace ilqgames {

class TimeInvariantConstraint : public Constraint {
 public:
  virtual ~TimeInvariantConstraint() {}

  // Check if this constraint is satisfied, and optionally return the value of a
  // function whose zero sub-level set corresponds to the feasible set.
  bool IsSatisfied(Time t, const VectorXf& input,
                   float* level = nullptr) const {
    return IsSatisfied(input, level);
  };
  virtual bool IsSatisfied(const VectorXf& input,
                           float* level = nullptr) const = 0;

  // Evaluate the barrier at the current input (use base class implementation
  // and provide arbitrary time).
  float Evaluate(const VectorXf& input) const {
    return Constraint::Evaluate(0.0, input);
  };

  // Quadraticize this cost at the given time and input, and add to the running
  // sum of gradients and Hessians.
  void Quadraticize(Time t, const VectorXf& input, MatrixXf* hess,
                    VectorXf* grad) const {
    Quadraticize(input, hess, grad);
  };
  virtual void Quadraticize(const VectorXf& input, MatrixXf* hess,
                            VectorXf* grad) const = 0;

 protected:
  explicit TimeInvariantConstraint(const std::string& name = "")
      : Constraint(name) {}
};  //\class TimeInvariantConstraint

}  // namespace ilqgames

#endif
