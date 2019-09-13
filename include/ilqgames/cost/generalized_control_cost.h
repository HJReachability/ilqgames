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
// The GeneralizedControlCost and its descendants allow for
// state-dependent control costs as one encounters in feedback linearization.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_COST_GENERALIZED_CONTROL_COST_H
#define ILQGAMES_COST_GENERALIZED_CONTROL_COST_H

#include <ilqgames/utils/types.h>

#include <string>

namespace ilqgames {

class GeneralizedControlCost {
 public:
  virtual ~GeneralizedControlCost() {}

  // Evaluate this cost at the current time and inputs.
  virtual float Evaluate(Time t, const VectorXf& x,
                         const VectorXf& u) const = 0;

  // Quadraticize this cost at the given time and inputs, and add to the running
  // sum of state gradients and state/control Hessians (if non-null).
  virtual void Quadraticize(Time t, const VectorXf& x, const VectorXf& u,
                            MatrixXf* hess_u, MatrixXf* hess_x,
                            VectorXf* grad_x) const = 0;

  // Access the name of this cost.
  const std::string& Name() const { return name_; }

 protected:
  explicit GeneralizedControlCost(float weight, const std::string& name = "")
      : weight_(weight), name_(name) {}

  // Multiplicative weight associated to this cost.
  const float weight_;

  // Name associated to every cost.
  const std::string name_;
};  //\class GeneralizedControlCost

}  // namespace ilqgames

#endif
