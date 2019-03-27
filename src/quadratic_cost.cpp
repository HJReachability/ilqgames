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
// Quadratic cost in a particular (or all) dimension(s).
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/cost/quadratic_approximation.h>
#include <ilqgames/cost/quadratic_cost.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>

namespace ilqgames {

// Evaluate this cost at the current input.
float QuadraticCost::Evaluate(const VectorXf& input) const {
  CHECK_LT(dimension_, input.size());

  // If dimension non-negative, then just square the desired dimension.
  if (dimension_ >= 0)
    return 0.5 * weight_ * input(dimension_) * input(dimension_);

  // Otherwise, cost is squared 2-norm of entire input.
  return 0.5 * weight_ * input.squaredNorm();
}

// Quadraticize this cost at the given input, and add to the running
// set of quadraticizations.
void QuadraticCost::Quadraticize(const VectorXf& input,
                                 QuadraticApproximation* q) const {
  CHECK_LT(dimension_, input.size());
  CHECK_NOTNULL(q);

  // Check dimensions.
  CHECK_EQ(input.size(), q->Q.rows());
  CHECK_EQ(input.size(), q->Q.cols());
  CHECK_EQ(input.size(), q->l.size());

  // Handle single dimension case first.
  if (dimension_ >= 0) {
    q->l(dimension_) += weight_ * input(dimension_);
    q->Q(dimension_, dimension_) += weight_;
  }

  // Handle dimension < 0 case.
  else {
    q->l += weight_ * input;
    q->Q.diagonal() =
        q->Q.diagonal() + VectorXf::Constant(input.size(), weight_);
  }
}

}  // namespace ilqgames
