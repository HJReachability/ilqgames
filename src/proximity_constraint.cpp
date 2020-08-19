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
// Constraint on proximity between two pairs of state dimensions (representing
// 2D position of vehicles whose states have been concatenated). Can be oriented
// either `inside` or `outside`, i.e., can constrain the states to be close
// together or far apart (respectively).
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/constraint/proximity_constraint.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <string>
#include <utility>

namespace ilqgames {

bool ProximityConstraint::IsSatisfiedLevel(const VectorXf& input,
                                           float* level) const {
  const float dx = input(xidx1_) - input(xidx2_);
  const float dy = input(yidx1_) - input(yidx2_);
  const float delta_sq = dx * dx + dy * dy;

  // Sign corresponding to orientation of this constraint.
  const float sign = (inside_) ? -1.0 : 1.0;

  // Maybe populate level.
  *level = sign * (threshold_sq_ - delta_sq);

  return (inside_) ? delta_sq < threshold_sq_ : delta_sq > threshold_sq_;
}

void ProximityConstraint::Quadraticize(const VectorXf& input, MatrixXf* hess,
                                       VectorXf* grad) const {
  CHECK_NOTNULL(hess);
  CHECK_NOTNULL(grad);

  // Check dimensions.
  CHECK_EQ(input.size(), hess->rows());
  CHECK_EQ(input.size(), hess->cols());
  CHECK_EQ(input.size(), grad->size());

  // Compute Hessian and gradient.
  const float dx = input(xidx1_) - input(xidx2_);
  const float dy = input(yidx1_) - input(yidx2_);
  const float dx2 = dx * dx;
  const float dy2 = dy * dy;
  const float delta_sq = dx2 + dy2;

  const float grad_coeff = 2.0 / (threshold_sq_ - delta_sq);
  const float weighted_grad_coeff = weight_ * grad_coeff;
  (*grad)(xidx1_) += weighted_grad_coeff * dx;
  (*grad)(xidx2_) -= weighted_grad_coeff * dx;
  (*grad)(yidx1_) += weighted_grad_coeff * dy;
  (*grad)(yidx2_) -= weighted_grad_coeff * dy;

  const float hess_x1x1 = weighted_grad_coeff * (grad_coeff * dx2 + 1.0);
  (*hess)(xidx1_, xidx1_) += hess_x1x1;
  (*hess)(xidx1_, xidx2_) -= hess_x1x1;
  (*hess)(xidx2_, xidx1_) -= hess_x1x1;
  (*hess)(xidx2_, xidx2_) += hess_x1x1;

  const float hess_y1y1 = weighted_grad_coeff * (grad_coeff * dy2 + 1.0);
  (*hess)(yidx1_, yidx1_) += hess_y1y1;
  (*hess)(yidx1_, yidx2_) -= hess_y1y1;
  (*hess)(yidx2_, yidx1_) -= hess_y1y1;
  (*hess)(yidx2_, yidx2_) += hess_y1y1;

  const float hess_x1y1 = weighted_grad_coeff * grad_coeff * dx * dy;
  (*hess)(xidx1_, yidx1_) += hess_x1y1;
  (*hess)(yidx1_, xidx1_) += hess_x1y1;
  (*hess)(xidx1_, yidx2_) -= hess_x1y1;
  (*hess)(yidx1_, xidx2_) -= hess_x1y1;
  (*hess)(xidx2_, yidx1_) -= hess_x1y1;
  (*hess)(yidx2_, xidx1_) -= hess_x1y1;
  (*hess)(xidx2_, yidx2_) += hess_x1y1;
  (*hess)(yidx2_, xidx2_) += hess_x1y1;
}

}  // namespace ilqgames
