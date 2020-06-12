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
// Semiquadratic cost on distance from a polyline.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/cost/semiquadratic_polyline2_cost.h>
#include <ilqgames/cost/time_invariant_cost.h>
#include <ilqgames/geometry/polyline2.h>
#include <ilqgames/utils/types.h>

#include <tuple>

namespace ilqgames {

float SemiquadraticPolyline2Cost::Evaluate(const VectorXf& input) const {
  CHECK_LT(xidx_, input.size());
  CHECK_LT(yidx_, input.size());

  // Compute signed squared distance by finding closest point.
  float signed_squared_distance;
  bool is_endpoint;
  polyline_.ClosestPoint(Point2(input(xidx_), input(yidx_)), nullptr, nullptr,
                         &signed_squared_distance, &is_endpoint);
  if (is_endpoint) {
    // If the is_endpoint flag is raised, we return 0.0.
    return 0.0;
  }
  // Check which side we're on.
  if (!IsActive(signed_squared_distance)) return 0.0;

  // Handle orientation.
  const float signed_distance = sgn(signed_squared_distance) *
                                std::sqrt(std::abs(signed_squared_distance));
  const float diff = signed_distance - threshold_;
  return 0.5 * weight_ * diff * diff;
}

void SemiquadraticPolyline2Cost::Quadraticize(const VectorXf& input,
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

  float signed_squared_distance;
  bool is_vertex;
  bool is_endpoint;
  LineSegment2 segment(Point2(0.0, 0.0), Point2(1.0, 1.0));
  const Point2 closest_point =
      polyline_.ClosestPoint(current_position, &is_vertex, &segment,
                             &signed_squared_distance, &is_endpoint);

  // Check if cost is active.
  if (!IsActive(signed_squared_distance)) return;

  // First checks whether the closest point is an endpoint of the polyline
  if (is_endpoint) return;

  // Handle cases separately depending on whether or not closest point is
  // a vertex of the polyline.
  float ddx = weight_;
  float ddy = weight_;
  float dxdy = 0.0;

  float scaling = std::sqrt(std::abs(signed_squared_distance));
  scaling = (scaling - std::abs(threshold_)) / scaling;
  float dx = weight_ * scaling * (current_position.x() - closest_point.x());
  float dy = weight_ * scaling * (current_position.y() - closest_point.y());
  if (!is_vertex) {
    const Point2 relative = current_position - segment.FirstPoint();
    const Point2& unit_segment = segment.UnitDirection();

    // Handle Hessian first.
    ddx = weight_ * unit_segment.y() * unit_segment.y();
    ddy = weight_ * unit_segment.x() * unit_segment.x();

    const float cross_term = -weight_ * unit_segment.x() * unit_segment.y();
    dxdy = cross_term;
    dxdy = cross_term;

    // Handle gradient.
    const float w_cross =
        weight_ * (relative.x() * unit_segment.y() -
                   relative.y() * unit_segment.x() - threshold_);

    dx = w_cross * unit_segment.y();
    dy = -w_cross * unit_segment.x();
  }

  ModifyDerivatives(input, &dx, &ddx, &dy, &ddy, &dxdy);

  (*grad)(xidx_) += dx;
  (*grad)(yidx_) += dy;

  (*hess)(xidx_, xidx_) += ddx;
  (*hess)(yidx_, yidx_) += ddy;
  (*hess)(xidx_, yidx_) += dxdy;
  (*hess)(yidx_, xidx_) += dxdy;
}

}  // namespace ilqgames
