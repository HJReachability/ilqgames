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
// Nominal value minus distance between two points in the given dimensions.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/cost/signed_distance_cost.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <numeric>

namespace ilqgames {

float SignedDistanceCost::Evaluate(const VectorXf& input) const {
  CHECK_LT(xdim1_, input.size());
  CHECK_LT(ydim1_, input.size());
  CHECK_LT(xdim2_, input.size());
  CHECK_LT(ydim2_, input.size());

  // Otherwise, cost is squared 2-norm of entire input.
  const float dx = input(xdim1_) - input(xdim2_);
  const float dy = input(ydim1_) - input(ydim2_);
  const float cost = nominal_ - std::hypot(dx, dy);

  return (less_is_positive_) ? cost : -cost;
}

void SignedDistanceCost::Quadraticize(const VectorXf& input, MatrixXf* hess,
                                      VectorXf* grad) const {
  CHECK_LT(xdim1_, input.size());
  CHECK_LT(ydim1_, input.size());
  CHECK_LT(xdim2_, input.size());
  CHECK_LT(ydim2_, input.size());
  CHECK_NOTNULL(hess);
  CHECK_NOTNULL(grad);

  // Check dimensions.
  CHECK_EQ(input.size(), hess->rows());
  CHECK_EQ(input.size(), hess->cols());
  CHECK_EQ(input.size(), grad->size());

  // Compute gradient and Hessian.
  const float s = (less_is_positive_) ? 1.0 : -1.0;
  const float delta_x = input(xdim1_) - input(xdim2_);
  const float delta_y = input(ydim1_) - input(ydim2_);
  const float norm = std::hypot(delta_x, delta_y);
  const float norm_3 = norm * norm * norm;

  const float dx1 = -s * delta_x / norm;
  const float dy1 = -s * delta_y / norm;
  const float ddx1 = -s * delta_y * delta_y / norm_3;
  const float ddy1 = -s * delta_x * delta_x / norm_3;
  const float dx1dy1 = s * delta_x * delta_y / norm_3;

  (*grad)(xdim1_) += dx1;
  (*grad)(ydim1_) += dy1;
  (*grad)(xdim2_) -= dx1;
  (*grad)(ydim2_) -= dy1;

  (*hess)(xdim1_, xdim1_) += ddx1;
  (*hess)(ydim1_, ydim1_) += ddy1;
  (*hess)(xdim1_, ydim1_) += dx1dy1;
  (*hess)(ydim1_, xdim1_) += dx1dy1;
  (*hess)(xdim2_, xdim2_) += ddx1;
  (*hess)(ydim2_, ydim2_) += ddy1;
  (*hess)(xdim2_, ydim2_) += dx1dy1;
  (*hess)(ydim2_, xdim2_) += dx1dy1;
  (*hess)(xdim1_, xdim2_) -= ddx1;
  (*hess)(xdim1_, ydim2_) -= dx1dy1;
  (*hess)(ydim1_, xdim2_) -= dx1dy1;
  (*hess)(ydim1_, ydim2_) -= ddy1;
  (*hess)(xdim2_, xdim1_) -= ddx1;
  (*hess)(xdim2_, ydim1_) -= dx1dy1;
  (*hess)(ydim2_, xdim1_) -= dx1dy1;
  (*hess)(ydim2_, ydim1_) -= ddy1;
}

}  // namespace ilqgames
