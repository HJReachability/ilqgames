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
// Line segment in 2D.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/geometry/line_segment2.h>
#include <ilqgames/utils/types.h>

namespace ilqgames {

bool LineSegment2::Side(const Point2& query) const {
  const Point2 relative_query = query - p1_;
  const float cross_product = relative_query.x() * unit_direction_.y() -
                              unit_direction_.x() * relative_query.y();

  return cross_product > 0.0;
}

Point2 LineSegment2::ClosestPoint(const Point2& query, bool* is_endpoint,
                                  float* signed_squared_distance) const {
  // Find query relative to p1.
  const Point2 relative_query = query - p1_;

  // Find dot product and signed length of cross product.
  const float dot_product = relative_query.dot(unit_direction_);
  const float cross_product = relative_query.x() * unit_direction_.y() -
                              unit_direction_.x() * relative_query.y();

  const float cross_product_sign = sgn(cross_product);

  // Determine closest point. This will either be an endpoint or the interior of
  // the segment.
  if (dot_product < 0.0) {
    // Query lies behind this line segment, so closest point is p1.
    if (is_endpoint) *is_endpoint = true;

    if (signed_squared_distance) {
      *signed_squared_distance =
          cross_product_sign * relative_query.squaredNorm();
    }

    return p1_;
  } else if (dot_product > length_) {
    // Closest point is p2.
    if (is_endpoint) *is_endpoint = true;

    if (signed_squared_distance) {
      *signed_squared_distance =
          cross_product_sign * (query - p2_).squaredNorm();
    }

    return p2_;
  }

  // Closest point is in the interior of the line segment.
  if (is_endpoint) *is_endpoint = false;

  if (signed_squared_distance)
    *signed_squared_distance =
        cross_product_sign * cross_product * cross_product;

  return p1_ + dot_product * unit_direction_;
}

}  // namespace ilqgames
