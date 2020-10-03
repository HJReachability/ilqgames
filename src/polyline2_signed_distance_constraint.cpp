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
  bool is_vertex = false;
  LineSegment2 closest_segment(Point2::Zero(), Point2::Zero());
  const Point2 closest_point = polyline_.ClosestPoint(
      Point2(input(xidx_), input(yidx_)), &is_vertex, &closest_segment);

  if (is_vertex)
    QuadraticizeVertex(t, input, hess, grad, closest_point);
  else
    QuadraticizeInterior(t, input, hess, grad, closest_point, closest_segment);
}

void Polyline2SignedDistanceConstraint::QuadraticizeVertex(
    Time t, const VectorXf& input, MatrixXf* hess, VectorXf* grad,
    const Point2& closest_point) const {
  // TODO!
}

void Polyline2SignedDistanceConstraint::QuadraticizeInterior(
    Time t, const VectorXf& input, MatrixXf* hess, VectorXf* grad,
    const Point2& closest_point, const LineSegment2& closest_segment) const {
  // TODO!
}

}  // namespace ilqgames
