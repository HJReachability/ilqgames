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
// Distance between two state positions.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/cost/relative_distance_cost.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>

namespace ilqgames {

float RelativeDistanceCost::Evaluate(const VectorXf& input) const {
  const float diff_x = input(dims1_.first) - input(dims2_.first);
  const float diff_y = input(dims1_.second) - input(dims2_.second);
  return weight_ * std::hypot(diff_x, diff_y);
}

void RelativeDistanceCost::Quadraticize(const VectorXf& input, MatrixXf* hess,
                                        VectorXf* grad) const {
  CHECK_NOTNULL(hess);

  // Check dimensions.
  CHECK_EQ(input.size(), hess->rows());
  CHECK_EQ(input.size(), hess->cols());

  if (grad) CHECK_EQ(input.size(), grad->size());

  const float diff_x = input(dims1_.first) - input(dims2_.first);
  const float diff_y = input(dims1_.second) - input(dims2_.second);
  const float dist = std::hypot(diff_x, diff_y);
  const float dist_3 = dist * dist * dist;

  const float ddx = weight_ * diff_y * diff_y / dist_3;
  const float ddy = weight_ * diff_x * diff_x / dist_3;
  const float dxdy = -weight_ * diff_x * diff_y / dist_3;

  (*hess)(dims1_.first, dims1_.first) += ddx;
  (*hess)(dims1_.first, dims1_.second) += dxdy;
  (*hess)(dims1_.second, dims1_.first) += dxdy;
  (*hess)(dims1_.second, dims1_.second) += ddy;

  (*hess)(dims2_.first, dims2_.first) += ddx;
  (*hess)(dims2_.first, dims2_.second) += dxdy;
  (*hess)(dims2_.second, dims2_.first) += dxdy;
  (*hess)(dims2_.second, dims2_.second) += ddy;

  (*hess)(dims1_.first, dims2_.first) -= ddx;
  (*hess)(dims1_.first, dims2_.second) -= dxdy;
  (*hess)(dims1_.second, dims2_.first) -= dxdy;
  (*hess)(dims1_.second, dims2_.second) -= ddy;

  (*hess)(dims2_.first, dims1_.first) -= ddx;
  (*hess)(dims2_.first, dims1_.second) -= dxdy;
  (*hess)(dims2_.second, dims1_.first) -= dxdy;
  (*hess)(dims2_.second, dims1_.second) -= ddy;

  if (grad) {
    const float dx = weight_ * diff_x / dist;
    const float dy = weight_ * diff_y / dist;

    (*grad)(dims1_.first) += dx;
    (*grad)(dims1_.second) += dy;
    (*grad)(dims2_.first) -= dx;
    (*grad)(dims2_.second) -= dy;
  }
}

}  // namespace ilqgames
