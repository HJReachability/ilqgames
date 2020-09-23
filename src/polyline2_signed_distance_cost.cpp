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
// Signed distance from a given polyline.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/cost/polyline2_signed_distance_cost.h>
#include <ilqgames/cost/time_invariant_cost.h>
#include <ilqgames/geometry/polyline2.h>
#include <ilqgames/utils/types.h>

#include <tuple>

namespace ilqgames {

float Polyline2SignedDistanceCost::Evaluate(const VectorXf& input) const {
  CHECK_LT(xidx_, input.size());
  CHECK_LT(yidx_, input.size());

  // Compute signed squared distance by finding closest point.
  float signed_squared_distance;
  polyline_.ClosestPoint(Point2(input(xidx_), input(yidx_)), nullptr, nullptr,
                         &signed_squared_distance);
  if (!oriented_same_as_polyline_) signed_squared_distance *= -1.0;

  return sgn(signed_squared_distance) *
             std::sqrt(std::abs(signed_squared_distance)) -
         nominal_;
}

void Polyline2SignedDistanceCost::Quadraticize(const VectorXf& input,
                                               MatrixXf* hess,
                                               VectorXf* grad) const {
  CHECK_LT(xidx_, input.size());
  CHECK_LT(yidx_, input.size());

  CHECK_NOTNULL(hess);
  CHECK_NOTNULL(grad);
  CHECK_EQ(input.size(), hess->rows());
  CHECK_EQ(input.size(), hess->cols());
  CHECK_EQ(input.size(), grad->size());

  // Unpack current position and find closest point / segment.
  const Point2 current_position(input(xidx_), input(yidx_));

  bool is_vertex;
  float signed_squared_distance;
  LineSegment2 segment(Point2(0.0, 0.0), Point2(1.0, 1.0));
  const Point2 closest_point = polyline_.ClosestPoint(
      current_position, &is_vertex, &segment, &signed_squared_distance);
  if (!oriented_same_as_polyline_) signed_squared_distance *= -1.0;

  const float sign = sgn(signed_squared_distance);
  const float distance = std::sqrt(std::abs(signed_squared_distance));
  const float delta_x = current_position.x() - closest_point.x();
  const float delta_y = current_position.y() - closest_point.y();

  // Handle cases separately depending on whether or not closest point is
  // a vertex of the polyline.
  float dx = sign * delta_x / distance;
  float dy = sign * delta_y / distance;

  const float denom = signed_squared_distance * distance;
  float ddx = delta_y * delta_y / denom;
  float ddy = delta_x * delta_x / denom;
  float dxdy = -delta_x * delta_y / denom;

  if (!is_vertex) {
    const Point2& unit_segment = segment.UnitDirection();

    dx = unit_segment.y();
    dy = -unit_segment.x();
    ddx = 0.0;
    ddy = 0.0;
    dxdy = 0.0;
  }

  (*grad)(xidx_) += dx;
  (*grad)(yidx_) += dy;

  (*hess)(xidx_, xidx_) += ddx;
  (*hess)(yidx_, yidx_) += ddy;
  (*hess)(xidx_, yidx_) += dxdy;
  (*hess)(yidx_, xidx_) += dxdy;
}

}  // namespace ilqgames
