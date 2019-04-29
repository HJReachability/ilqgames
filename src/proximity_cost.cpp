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
// Penalizes 1.0 / (relative distance)^2 between two pairs of state dimensions
// (representing two positions of vehicles whose states have been concatenated).
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/cost/proximity_cost.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>

namespace ilqgames {

float ProximityCost::Evaluate(const VectorXf& input) const {
  const float dx = input(xidx1_) - input(xidx2_);
  const float dy = input(yidx1_) - input(yidx2_);
  return 0.5 * weight_ / (dx * dx + dy * dy);
}

void ProximityCost::Quadraticize(const VectorXf& input, MatrixXf* hess,
                                 VectorXf* grad) const {
  CHECK_NOTNULL(hess);

  // Check dimensions.
  CHECK_EQ(input.size(), hess->rows());
  CHECK_EQ(input.size(), hess->cols());

  if (grad) CHECK_EQ(input.size(), grad->size());

  // Compute Hessian and gradient.
  const float dx = input(xidx1_) - input(xidx2_);
  const float dy = input(yidx1_) - input(yidx2_);
  const float distance_sq = dx * dx + dy * dy;
  const float weight_over_distance_4th = weight_ / (distance_sq * distance_sq);

  const float hess_x1x1 =
      weight_over_distance_4th * (4.0 * dx * dx / distance_sq - 1.0);
  (*hess)(xidx1_, xidx1_) += hess_x1x1;
  (*hess)(xidx1_, xidx2_) -= hess_x1x1;
  (*hess)(xidx2_, xidx1_) -= hess_x1x1;
  (*hess)(xidx2_, xidx2_) += hess_x1x1;

  const float hess_y1y1 =
      weight_over_distance_4th * (4.0 * dy * dy / distance_sq - 1.0);
  (*hess)(yidx1_, yidx1_) += hess_y1y1;
  (*hess)(yidx1_, yidx2_) -= hess_y1y1;
  (*hess)(yidx2_, yidx1_) -= hess_y1y1;
  (*hess)(yidx2_, yidx2_) += hess_y1y1;

  const float hess_x1y1 =
      4.0 * weight_over_distance_4th * dx * dy / distance_sq;
  (*hess)(xidx1_, yidx1_) += hess_x1y1;
  (*hess)(yidx1_, xidx1_) += hess_x1y1;

  (*hess)(xidx1_, yidx2_) -= hess_x1y1;
  (*hess)(yidx2_, xidx1_) -= hess_x1y1;

  (*hess)(xidx2_, yidx1_) -= hess_x1y1;
  (*hess)(yidx1_, xidx2_) -= hess_x1y1;

  (*hess)(xidx2_, yidx2_) += hess_x1y1;
  (*hess)(yidx2_, xidx2_) += hess_x1y1;

  if (grad) {
    const float ddx1 = -weight_over_distance_4th * dx;
    (*grad)(xidx1_) += ddx1;
    (*grad)(xidx2_) -= ddx1;

    const float ddy1 = -weight_over_distance_4th * dy;
    (*grad)(yidx1_) += ddy1;
    (*grad)(yidx2_) -= ddy1;
  }
}

}  // namespace ilqgames
