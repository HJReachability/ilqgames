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
// Base class for all constraints. We assume that all constraints are
// *inequalities*, which support a check for satisfaction. All constraints must
// also implement the cost interface corresponding to a barrier function.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_CONSTRAINT_CONSTRAINT_H
#define ILQGAMES_CONSTRAINT_CONSTRAINT_H

#include <ilqgames/cost/cost.h>
#include <ilqgames/utils/types.h>

#include <string>

namespace ilqgames {

class Constraint : public Cost {
 public:
  virtual ~Constraint() {}

  // Set or multiplicatively scale the barrier weight. This will typically
  // decrease with successive solves in order to improve the approximation of
  // the barrier-free objective.
  void SetBarrierWeight(float weight) { weight_ = weight; }
  void ScaleBarrierWeight(float scale) { weight_ *= scale; }

  // Check if this constraint is satisfied, and optionally return the value of a
  // function whose zero sub-level set corresponds to the feasible set.
  virtual bool IsSatisfied(Time t, const VectorXf& input,
                           float* level = nullptr) const = 0;

  // Evaluate the barrier at the current time and input.
  virtual float Evaluate(Time t, const VectorXf& input) const = 0;

  // Quadraticize the barrier at the given time and input, and add to the
  // running sum of gradients and Hessians (if non-null).
  virtual void Quadraticize(Time t, const VectorXf& input, MatrixXf* hess,
                            VectorXf* grad) const = 0;

  // Access the name of this cost.
  const std::string& Name() const { return name_; }

 protected:
  explicit Constraint(const std::string& name = "") : Cost(1.0, name) {}
};  //\class Constraint

}  // namespace ilqgames

#endif
