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
// Base class for all barrier constraints. We assume that all constraints are
// *inequalities*, which support a check for satisfaction. All constraints must
// also implement the cost interface corresponding to a barrier function.
// Further, all constraints also must have a corresponding cost associated to
// them which, unlike a log barrier which *forces* iterates to remain feasible,
// merely *encourages* iterates to become feasible. This is useful, e.g., when
// initial guesses are not feasible.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_CONSTRAINT_BARRIER_BARRIER_H
#define ILQGAMES_CONSTRAINT_BARRIER_BARRIER_H

#include <ilqgames/cost/cost.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <memory>
#include <string>

namespace ilqgames {

class Barrier : public Cost {
 public:
  virtual ~Barrier() {}

  // Set or multiplicatively scale the barrier weight. This will typically
  // decrease with successive solves in order to improve the approximation of
  // the barrier-free objective.
  void ResetBarrierWeight() {
    weight_ = kInitialBarrierWeight;
    if (equivalent_cost_.get())
      equivalent_cost_->SetWeight(kInitialEquivalentCostWeight);
  }
  void ScaleBarrierWeight(float scale) {
    weight_ *= scale;
    if (equivalent_cost_.get()) equivalent_cost_->ScaleWeight(scale);
  }

  // Check if this constraint is satisfied, and optionally return the value of a
  // function whose zero sub-level set corresponds to the feasible set.
  virtual bool IsSatisfiedLevel(Time t, const VectorXf& input,
                                float* level) const = 0;
  bool IsSatisfied(Time t, const VectorXf& input) const {
    float level;
    return IsSatisfiedLevel(t, input, &level);
  }

  // Evaluate the barrier at the current time and input.
  float Evaluate(Time t, const VectorXf& input) const;

  // Quadraticize the barrier at the given time and input, and add to the
  // running sum of gradients and Hessians (if non-null).
  virtual void Quadraticize(Time t, const VectorXf& input, MatrixXf* hess,
                            VectorXf* grad) const = 0;

  // Accessors.
  const Cost& EquivalentCost() const {
    CHECK_NOTNULL(equivalent_cost_.get());
    return *equivalent_cost_;
  }

 protected:
  explicit Barrier(const std::string& name = "")
      : Cost(kInitialBarrierWeight, name) {}

  // "Equivalent" well-defined cost to encourage constraint satisfaction, e.g.,
  // when an initial iterate is infeasible.
  std::unique_ptr<Cost> equivalent_cost_;
  static constexpr float kInitialBarrierWeight = 1.0;
  static constexpr float kInitialEquivalentCostWeight = 0.1;
  static constexpr float kCostBuffer = 1.0;
};  //\class Barrier

}  // namespace ilqgames

#endif
