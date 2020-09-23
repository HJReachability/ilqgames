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
// Base class for all time-invariant cost functions.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_COST_TIME_INVARIANT_COST_H
#define ILQGAMES_COST_TIME_INVARIANT_COST_H

#include <ilqgames/cost/cost.h>
#include <ilqgames/utils/types.h>

#include <string>

namespace ilqgames {

class TimeInvariantCost : public Cost {
 public:
  virtual ~TimeInvariantCost() {}

  // Evaluate this cost at the given input.
  virtual float Evaluate(const VectorXf& input) const = 0;
  float Evaluate(Time t, const VectorXf& input) const {
    return Evaluate(input);
  }

  // Quadraticize this cost at the given input, and add to the running set of
  // sum of gradients and Hessians.
  virtual void Quadraticize(const VectorXf& input, MatrixXf* hess,
                            VectorXf* grad) const = 0;
  void Quadraticize(Time t, const VectorXf& input, MatrixXf* hess,
                    VectorXf* grad) const {
    Quadraticize(input, hess, grad);
  }

 protected:
  explicit TimeInvariantCost(float weight, const std::string& name)
      : Cost(weight, name) {}
};  //\class TimeInvariantCost

}  // namespace ilqgames

#endif
