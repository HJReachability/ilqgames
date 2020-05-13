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
// Constraint on the signed distance to a polyline. Can be oriented either
// `right` or `left`, i.e., can constrain the signed distance to be either > or
// < the given threshold, respectively.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/constraint/polyline2_signed_distance_constraint.h>
#include <ilqgames/geometry/polyline2.h>
#include <ilqgames/utils/types.h>

#include <string>
#include <utility>

namespace ilqgames {

bool Polyline2SignedDistanceConstraint::IsSatisfied(const VectorXf& input,
                                                    float* level) const {
  CHECK_LT(xidx_, input.size());
  CHECK_LT(yidx_, input.size());

  // Compute signed squared distance by finding closest point.
  float signed_distance_sq;
  polyline_.ClosestPoint(Point2(input(xidx_), input(yidx_)), nullptr, nullptr,
                         &signed_distance_sq);

  // Maybe set level.
  const float sign = (oriented_right_) ? 1.0 : -1.0;
  if (level) *level = sign * (signed_threshold_sq_ - signed_distance_sq);

  return (oriented_right_) ? signed_distance_sq > signed_threshold_sq_
                           : signed_distance_sq < signed_threshold_sq_;
}

void Polyline2SignedDistanceConstraint::Quadraticize(const VectorXf& input,
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

  float signed_distance_sq;
  bool is_vertex;
  bool is_endpoint;
  LineSegment2 segment(Point2(0.0, 0.0), Point2(1.0, 1.0));
  const Point2 closest_point =
      polyline_.ClosestPoint(current_position, &is_vertex, &segment,
                             &signed_distance_sq, &is_endpoint);

  // Sign corresponding to orientation of this constraint and of the signed
  // distance itself.
  const float orientation = (oriented_right_) ? 1.0 : -1.0;
  const float sign = sgn(signed_distance_sq);

  // Barrier level.
  const float dx = current_position.x() - closest_point.x();
  const float dy = current_position.y() - closest_point.y();
  const float dx2 = dx * dx;
  const float dy2 = dy * dy;
  const float level = orientation * (signed_threshold_sq_ - signed_distance_sq);

  // Handle cases separately depending on whether or not closest point is
  // a vertex of the polyline.
  if (!is_vertex) {
    const Point2 relative = current_position - segment.FirstPoint();
    const Point2& unit_direction = segment.UnitDirection();
    const float cross =
        relative.x() * unit_direction.y() - relative.y() * unit_direction.x();
    CHECK_EQ(sgn(cross), sgn(signed_distance_sq));

    const float coeff = 2.0 * orientation * sign / level;
    const float grad_coeff = coeff * cross;
    const float weighted_grad_coeff = weight_ * grad_coeff;
    (*grad)(xidx_) += weighted_grad_coeff * unit_direction.y();
    (*grad)(yidx_) -= weighted_grad_coeff * unit_direction.x();

    const float hess_coeff = weight_ * coeff * (grad_coeff * cross + 1.0);
    const float hess_xy = hess_coeff * unit_direction.x() * unit_direction.y();
    (*hess)(xidx_, xidx_) +=
        hess_coeff * unit_direction.y() * unit_direction.y();
    (*hess)(yidx_, yidx_) +=
        hess_coeff * unit_direction.x() * unit_direction.x();
    (*hess)(xidx_, yidx_) -= hess_xy;
    (*hess)(yidx_, xidx_) -= hess_xy;
  } else {
    // Closest point is a vertex.
    const float grad_coeff = 2.0 * orientation * sign / level;
    const float weighted_grad_coeff = weight_ * grad_coeff;
    (*grad)(xidx_) += weighted_grad_coeff * dx;
    (*grad)(yidx_) += weighted_grad_coeff * dy;

    (*hess)(xidx_, xidx_) += weighted_grad_coeff * (grad_coeff * dx2 + 1.0);
    (*hess)(yidx_, yidx_) += weighted_grad_coeff * (grad_coeff * dy2 + 1.0);

    const float hess_xy = weighted_grad_coeff * grad_coeff * dx * dy;
    (*hess)(xidx_, yidx_) += hess_xy;
    (*hess)(yidx_, xidx_) += hess_xy;
  }
}  // namespace ilqgames

}  // namespace ilqgames
