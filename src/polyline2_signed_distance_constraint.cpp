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
// (Time-invariant) inequality constraint encoding
//           g(x) = (+/-) (signed_distance(x, polyline) - d) <= 0
//
// NOTE: The `keep_left` argument specifies the sign of g (true corresponds to
// positive).

///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/constraint/polyline2_signed_distance_constraint.h>
#include <ilqgames/geometry/line_segment2.h>
#include <ilqgames/geometry/polyline2.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <memory>
#include <string>

namespace ilqgames {

float Polyline2SignedDistanceConstraint::Evaluate(const VectorXf& input) const {
  CHECK_LT(xidx_, input.size());
  CHECK_LT(yidx_, input.size());

  // Compute signed squared distance by finding closest point.
  float signed_distance_sq;
  polyline_.ClosestPoint(Point2(input(xidx_), input(yidx_)), nullptr, nullptr,
                         &signed_distance_sq);

  const float value = signed_sqrt(signed_distance_sq) - threshold_;
  return (keep_left_) ? value : -value;
}

void Polyline2SignedDistanceConstraint::Quadraticize(Time t,
                                                     const VectorXf& input,
                                                     MatrixXf* hess,
                                                     VectorXf* grad) const {
  CHECK_LT(xidx_, input.size());
  CHECK_LT(yidx_, input.size());
  CHECK_NOTNULL(grad);
  CHECK_NOTNULL(hess);
  CHECK_EQ(hess->rows(), input.size());
  CHECK_EQ(hess->cols(), input.size());
  CHECK_EQ(grad->size(), input.size());

  // Find closest point/segment and whether closest point is an interior point
  // or a vertex.
  bool is_vertex;
  LineSegment2 closest_segment(Point2::Zero(), Point2::Zero());
  float signed_distance_sq;
  const Point2 closest_point =
      polyline_.ClosestPoint(Point2(input(xidx_), input(yidx_)), &is_vertex,
                             &closest_segment, &signed_distance_sq);

  // Compute value of g.
  const float g = (keep_left_) ? signed_sqrt(signed_distance_sq) - threshold_
                               : threshold_ - signed_sqrt(signed_distance_sq);

  if (is_vertex)
    QuadraticizeVertex(t, input, hess, grad, g, closest_point);
  else
    QuadraticizeInterior(t, input, hess, grad, g, closest_segment);
}

void Polyline2SignedDistanceConstraint::QuadraticizeVertex(
    Time t, const VectorXf& input, MatrixXf* hess, VectorXf* grad, float g,
    const Point2& closest_point) const {
  // Unpack geometry.
  const float x = input(xidx_);
  const float y = input(yidx_);
  const float px = closest_point.x();
  const float py = closest_point.y();
  const float rx = x - px;
  const float ry = y - py;
  const float d_sq = rx * rx + ry * ry;
  const float d = std::sqrt(d_sq);
  const float sign = (keep_left_) ? 1.0 : -1.0;

  /// Compute derivatives of g using symbolic differentiation.
  float dx = sign * rx / d;
  float ddx = sign * ry * ry / (d_sq * d);
  float dxdy = -sign * rx * ry / (d_sq * d);
  float dy = sign * ry * ry / (d_sq * d);
  float ddy = sign * rx * rx / (d_sq * d);
  ;

  // Modify derivatives according to augmented Lagrangian.
  ModifyDerivatives(t, g, &dx, &ddx, &dy, &ddy, &dxdy);

  // Populate grad and hess.
  (*grad)(xidx_) += dx;
  (*grad)(yidx_) += dy;

  (*hess)(xidx_, xidx_) += ddx;
  (*hess)(xidx_, yidx_) += dxdy;
  (*hess)(yidx_, xidx_) += dxdy;
  (*hess)(yidx_, yidx_) += ddy;
}

void Polyline2SignedDistanceConstraint::QuadraticizeInterior(
    Time t, const VectorXf& input, MatrixXf* hess, VectorXf* grad, float g,
    const LineSegment2& closest_segment) const {
  // Unpack geometry.
  const float x = input(xidx_);
  const float y = input(yidx_);
  const float px = closest_segment.FirstPoint().x();
  const float py = closest_segment.FirstPoint().y();
  const float rx = x - px;
  const float ry = y - py;
  const float d_sq = rx * rx + ry * ry;
  const float d = std::sqrt(d_sq);
  const float ux = closest_segment.UnitDirection().x();
  const float uy = closest_segment.UnitDirection().y();
  const float sign = (keep_left_) ? 1.0 : -1.0;

  // Compute derivatives of g using symbolic differentiation.
  float dx = sign *
             (uy * d_sq - px * px * uy - uy * x * x + px * py * ux +
              2 * px * uy * x - py * ux * x - px * ux * y + ux * x * y) /
             (d_sq * d);
  float ddx =
      sign *
      (-ry * (2 * ux * px * px + 3 * uy * px * py - 4 * ux * px * x -
              3 * uy * px * y - ux * py * py - 3 * uy * py * x +
              2 * ux * py * y + 2 * ux * x * x + 3 * uy * x * y - ux * y * y)) /
      (d_sq * d_sq * d);
  float dy = -sign *
             (ux * d_sq - py * py * ux - ux * y * y + px * py * uy -
              py * uy * x - px * uy * y + 2 * py * ux * y + uy * x * y) /
             (d_sq * d);
  float ddy =
      sign *
      (-rx * (uy * px * px - 3 * ux * px * py - 2 * uy * px * x +
              3 * ux * px * y - 2 * uy * py * py + 3 * ux * py * x +
              4 * uy * py * y + uy * x * x - 3 * ux * x * y - 2 * uy * y * y)) /
      (d_sq * d_sq * d);
  float dxdy =
      sign * (-uy * ry + ux * rx -
              (3 * rx * ry * (-uy * rx + ux * ry) / d_sq) / (d_sq * d));

  // Modify derivatives according to augmented Lagrangian.
  ModifyDerivatives(t, g, &dx, &ddx, &dy, &ddy, &dxdy);

  // Populate grad and hess.
  (*grad)(xidx_) += dx;
  (*grad)(yidx_) += dy;

  (*hess)(xidx_, xidx_) += ddx;
  (*hess)(xidx_, yidx_) += dxdy;
  (*hess)(yidx_, xidx_) += dxdy;
  (*hess)(yidx_, yidx_) += ddy;
}

}  // namespace ilqgames
