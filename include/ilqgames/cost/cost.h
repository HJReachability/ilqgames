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
// Base class for all cost functions. All costs must support evaluation and
// quadraticization. By default, cost functions are of only state or control.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_COST_COST_H
#define ILQGAMES_COST_COST_H

#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <string>

namespace ilqgames {

class Cost {
 public:
  virtual ~Cost() {}

  // Evaluate this cost at the current time and input. Also provide a utility
  // for evaluating the exponentiated version of this cost, i.e. exp(a * cost)
  virtual float Evaluate(Time t, const VectorXf& input) const = 0;
  virtual float EvaluateExponential(Time t, const VectorXf& input) const {
    return std::exp(exponential_constant_ * Evaluate(t, input));
  }

  // Quadraticize this cost at the given time and input, and add to the running
  // sum of gradients and Hessians (if non-null).
  virtual void Quadraticize(Time t, const VectorXf& input, MatrixXf* hess,
                            VectorXf* grad = nullptr) const = 0;

  // Access the name of this cost.
  const std::string& Name() const { return name_; }

  // Set exponential constant. See below for further details.
  void SetExponentialConstant(float a) {
    CHECK_GE(a, 0.0);
    exponential_constant_ = a;
  }
  bool IsExponentiated() const {
    return std::abs(exponential_constant_) > constants::kSmallNumber;
  }

  // Reset the initial time associated to this cost.
  static void ResetInitialTime(Time t0) { initial_time_ = t0; };

  // Reset and scale weight.
  void SetWeight(float weight) { weight_ = weight; }
  void ScaleWeight(float scale) { weight_ *= scale; }

 protected:
  explicit Cost(float weight, const std::string& name = "")
      : weight_(weight), name_(name), exponential_constant_(0.0) {}

  // Multiplicative weight associated to this cost.
  float weight_;

  // Name associated to every cost.
  const std::string name_;

  // Initial time associated to this cost.
  static Time initial_time_;

  // Constant multiplying all costs before they get exponentiated in
  // quadraticization.
  // NOTE: This defaults to zero but can be positive to get a max over time
  // approximation.
  // NOTE: This only takes effect during quadraticization, but not during cost
  // evaluation, since the intent is to approximate a min or max over time.
  // NOTE: When this is nearly zero, we just assume that exponentiated costs are
  // not active and resort to regular (non-exponentiated costs).
  float exponential_constant_;
};  //\class Cost

}  // namespace ilqgames

#endif
