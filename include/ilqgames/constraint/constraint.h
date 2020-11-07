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
// Base class for all explicit (scalar-valued) constraints. These
// constraints are of the form: g(x) = 0 or g(x) <= 0 for some vector x.
//
// In addition to checking for satisfaction (and returning the constraint value
// g(x)), they also support computing first and second derivatives of the
// constraint value itself and the square of the constraint value, each scaled
// by lambda or mu respectively (from the augmented Lagrangian). That is, they
// compute gradients and Hessians of
//         L(x, lambda, mu) = lambda * g(x) + mu * g(x) * g(x) / 2
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_CONSTRAINT_CONSTRAINT_H
#define ILQGAMES_CONSTRAINT_CONSTRAINT_H

#include <ilqgames/cost/cost.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <memory>
#include <string>

namespace ilqgames {

class Constraint : public Cost {
 public:
  virtual ~Constraint() {}

  // Check if this constraint is satisfied, and optionally return the constraint
  // value, which equals zero if the constraint is satisfied.
  bool IsSatisfied(Time t, const VectorXf& input, float* level) const {
    const float value = Evaluate(t, input);
    if (level) *level = value;

    return IsSatisfied(value);
  }
  bool IsSatisfied(float level) const {
    return (is_equality_) ? std::abs(level) <= constants::kSmallNumber
                          : level <= constants::kSmallNumber;
  }

  // Evaluate this constraint value, i.e., g(x), and the augmented Lagrangian,
  // i.e., lambda g(x) + mu g(x) g(x) / 2.
  virtual float Evaluate(Time t, const VectorXf& input) const = 0;
  float EvaluateAugmentedLagrangian(Time t, const VectorXf& input) const {
    const float g = Evaluate(t, input);
    const float lambda = lambdas_[TimeIndex(t)];
    return lambda * g + 0.5 * Mu(lambda, g) * g * g;
  }

  // Quadraticize the constraint value and its square, each scaled by lambda or
  // mu, respectively (terms in the augmented Lagrangian).
  virtual void Quadraticize(Time t, const VectorXf& input, MatrixXf* hess,
                            VectorXf* grad) const = 0;

  // Accessors and setters.
  bool IsEquality() const { return is_equality_; }
  float& Lambda(Time t) { return lambdas_[TimeIndex(t)]; }
  float Lambda(Time t) const { return lambdas_[TimeIndex(t)]; }
  void IncrementLambda(Time t, float value) {
    const size_t kk = TimeIndex(t);
    const float new_lambda = lambdas_[kk] + mu_ * value;
    lambdas_[kk] = (is_equality_) ? new_lambda : std::max(0.0f, new_lambda);
  }
  void ScaleLambdas(float scale) {
    for (auto& lambda : lambdas_) lambda *= scale;
  }
  static float& GlobalMu() { return mu_; }
  static void ScaleMu(float scale) { mu_ *= scale; }
  float Mu(Time t, const VectorXf& input) const {
    const float g = Evaluate(t, input);
    return Mu(Lambda(t), g);
  }
  float Mu(float lambda, float g) const {
    if (!is_equality_ && g <= constants::kSmallNumber &&
        std::abs(lambda) <= constants::kSmallNumber)
      return 0.0;
    return mu_;
  }

 protected:
  explicit Constraint(bool is_equality, const std::string& name)
      : Cost(1.0, name),
        is_equality_(is_equality),
        lambdas_(time::kNumTimeSteps, constants::kDefaultLambda) {}

  // Modify derivatives to account for the multipliers and the quadratic term in
  // the augmented Lagrangian. The inputs are the derivatives of g in the
  // appropriate variables (assumed to be arbitrary coordinates of the input,
  // here called x and y).
  void ModifyDerivatives(Time t, float g, float* dx, float* ddx,
                         float* dy = nullptr, float* ddy = nullptr,
                         float* dxdy = nullptr) const;

  // Is this an equality constraint? If not, it is an inequality constraint.
  bool is_equality_;

  // Multipliers, one per time step. Also a static augmented multiplier for an
  // augmented Lagrangian.
  std::vector<float> lambdas_;
  static float mu_;
};  //\class Constraint

}  // namespace ilqgames

#endif
