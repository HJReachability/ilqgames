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
// Quadratic penalty on distance from where we should be along a given polyline
// if we were traveling at the given nominal speed.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/cost/route_progress_cost.h>
#include <ilqgames/geometry/polyline2.h>
#include <ilqgames/utils/types.h>

#include <string>
#include <tuple>

namespace ilqgames {

float RouteProgressCost::Evaluate(Time t, const VectorXf& input) const {
  CHECK_LT(xidx_, input.size());
  CHECK_LT(yidx_, input.size());

  const float desired_route_pos =
      initial_route_pos_ + (t - initial_time_) * nominal_speed_;
  const Point2 desired = polyline_.PointAt(desired_route_pos, nullptr, nullptr);

  const float dx = input(xidx_) - desired.x();
  const float dy = input(yidx_) - desired.y();
  return 0.5 * weight_ * (dx * dx + dy * dy);
}

void RouteProgressCost::Quadraticize(Time t, const VectorXf& input,
                                     MatrixXf* hess, VectorXf* grad) const {
  CHECK_LT(xidx_, input.size());
  CHECK_LT(yidx_, input.size());

  CHECK_NOTNULL(hess);
  CHECK_NOTNULL(grad);
  CHECK_EQ(input.size(), hess->rows());
  CHECK_EQ(input.size(), hess->cols());
  CHECK_EQ(input.size(), grad->size());

  // Unpack current position and find closest point / segment.
  const Point2 current_position(input(xidx_), input(yidx_));
  const float desired_route_pos =
      initial_route_pos_ + (t - initial_time_) * nominal_speed_;

  bool is_endpoint;
  const Point2 route_point =
      polyline_.PointAt(desired_route_pos, nullptr, nullptr, &is_endpoint);

  // Compute gradient and Hessian.
  const float diff_x = current_position.x() - route_point.x();
  const float diff_y = current_position.y() - route_point.y();
  float dx = weight_ * diff_x;
  float dy = weight_ * diff_y;
  float ddx = weight_;
  float ddy = weight_;
  float dxdy = 0.0;

  ModifyDerivatives(t, input, &dx, &ddx, &dy, &ddy, &dxdy);

  (*grad)(xidx_) += dx;
  (*grad)(yidx_) += dy;

  (*hess)(xidx_, xidx_) += ddx;
  (*hess)(yidx_, yidx_) += ddy;
  (*hess)(xidx_, yidx_) += dxdy;
  (*hess)(yidx_, xidx_) += dxdy;
}

}  // namespace ilqgames
