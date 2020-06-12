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
// Penalizes min(I(abs(x1 - x2) < d) * (d - abs(x1 - x2))^2,
//               I(abs(y1 - y2) < d) * (d - abs(y1 - y2))^2).
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/cost/locally_convex_proximity_cost.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>

namespace ilqgames {

float LocallyConvexProximityCost::Evaluate(const VectorXf& input) const {
  const float dx = input(xidx1_) - input(xidx2_);
  const float dy = input(yidx1_) - input(yidx2_);

  if (dx * dx >= threshold_sq_ || dy * dy >= threshold_sq_) return 0.0;

  const float delta_x = threshold_ - std::abs(dx);
  const float delta_y = threshold_ - std::abs(dy);
  return 0.5 * weight_ * std::min(delta_x * delta_x, delta_y * delta_y);
}

void LocallyConvexProximityCost::Quadraticize(const VectorXf& input,
                                              MatrixXf* hess,
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

  if (dx * dx >= threshold_sq_ || dy * dy >= threshold_sq_) return;

  // Which dimension is active, x or y?
  const float delta_x = threshold_ - std::abs(dx);
  const float delta_y = threshold_ - std::abs(dy);

  const bool is_x_active = delta_x * delta_x < delta_y * delta_y;

  if (is_x_active) {
    float dx1 = -weight_ * delta_x;
    float ddx1 = weight_;
    ModifyDerivatives(input, &dx1, &ddx1);

    (*grad)(xidx1_) += dx1;
    (*grad)(xidx2_) -= dx1;

    (*hess)(xidx1_, xidx1_) += ddx1;
    (*hess)(xidx2_, xidx2_) += ddx1;
    (*hess)(xidx1_, xidx2_) -= ddx1;
    (*hess)(xidx2_, xidx1_) -= ddx1;
  } else {
    float dy1 = -weight_ * delta_y;
    float ddy1 = weight_;
    ModifyDerivatives(input, &dy1, &ddy1);

    (*grad)(yidx1_) += dy1;
    (*grad)(yidx2_) -= dy1;

    (*hess)(yidx1_, yidx1_) += ddy1;
    (*hess)(yidx2_, yidx2_) += ddy1;
    (*hess)(yidx1_, yidx2_) -= ddy1;
    (*hess)(yidx2_, yidx1_) -= ddy1;
  }
}

}  // namespace ilqgames
