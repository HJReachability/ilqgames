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
// Shape-creation utilities based on polylines.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/geometry/draw_shapes.h>
#include <ilqgames/geometry/polyline2.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>

namespace ilqgames {

Polyline2 DrawSquare(const Point2& center, float side_length) {
  CHECK_GT(side_length, 0.0);
  const float half_side = 0.5 * side_length;
  return Polyline2({center + Point2(half_side, half_side),
                    center + Point2(-half_side, half_side),
                    center + Point2(-half_side, -half_side),
                    center + Point2(half_side, -half_side),
                    center + Point2(half_side, half_side)});
}

Polyline2 DrawCircle(const Point2& center, float radius, size_t num_segments) {
  CHECK_GT(radius, 0.0);

  PointList2 poly;
  poly.push_back(center + Point2(radius, 0.0));
  for (size_t ii = 0; ii < num_segments; ii++) {
    const float angle = 2.0 * M_PI * static_cast<float>(ii + 1) /
                        static_cast<float>(num_segments);
    poly.push_back(center + radius * Point2(std::cos(angle), std::sin(angle)));
  }

  return Polyline2(poly);
}

}  // namespace ilqgames
