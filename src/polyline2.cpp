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
// Polyline2 class for piecewise linear paths in 2D.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/geometry/line_segment2.h>
#include <ilqgames/geometry/polyline2.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>

namespace ilqgames {

Polyline2::Polyline2(const PointList2& points) : length_(0.0) {
  CHECK_GT(points.size(), 1);
  cumulative_lengths_.push_back(length_);

  // Parse into list of line segents.
  for (size_t ii = 1; ii < points.size(); ii++) {
    segments_.emplace_back(points[ii - 1], points[ii]);
    length_ += segments_.back().Length();
    cumulative_lengths_.push_back(length_);
  }
}

void Polyline2::AddPoint(const Point2& point) {
  segments_.emplace_back(segments_.back().SecondPoint(), point);
  length_ += segments_.back().Length();
}

Point2 Polyline2::PointAt(float route_pos, bool* is_vertex,
                          LineSegment2* segment, bool* is_endpoint) const {
  auto upper = std::upper_bound(cumulative_lengths_.begin(),
                                cumulative_lengths_.end(), route_pos);
  if (upper == cumulative_lengths_.end()) {
    LOG(WARNING) << "Route position " << route_pos
                 << " was off the end of the route.";
    upper--;
  }

  // Find the index of the line segment which contains this route position.
  upper--;
  const size_t idx = std::distance(cumulative_lengths_.begin(), upper);
  if (segment) *segment = segments_[idx];

  // Walk along this line segment the remaining distance.
  const float remaining = route_pos - cumulative_lengths_[idx];
  CHECK_GE(remaining, 0.0);

  if (is_vertex) {
    *is_vertex = remaining < constants::kSmallNumber ||
                 remaining > segments_[idx].Length();
  }

  const Point2 return_point =
      segments_[idx].FirstPoint() + remaining * segments_[idx].UnitDirection();
  if (is_endpoint) {
    if (idx != 0 && idx != segments_.size() - 1)
      *is_endpoint = false;
    else
      *is_endpoint = (return_point == segments_.front().FirstPoint()) ||
                     (return_point == segments_.back().SecondPoint());
  }

  return return_point;
}

Point2 Polyline2::ClosestPoint(const Point2& query, bool* is_vertex,
                               LineSegment2* segment,
                               float* signed_squared_distance,
                               bool* is_endpoint) const {
  // Walk along each line segment and remember which was closest.
  float closest_signed_squared_distance = constants::kInfinity;
  Point2 closest_point;

  float current_signed_squared_distance;
  int segment_idx = 0;
  int segment_counter = 0;
  bool is_segment_endpoint;
  for (const auto& s : segments_) {
    const Point2 current_point = s.ClosestPoint(
        query, &is_segment_endpoint, &current_signed_squared_distance);

    if (std::abs(current_signed_squared_distance) <
        std::abs(closest_signed_squared_distance)) {
      // If this is an endpoint, compute which side of the polyline this is on
      // by finding which side of the line segment from the previous point to
      // the next point this is on.
      if (is_segment_endpoint &&
          (segment_counter > 0 || current_point == s.SecondPoint()) &&
          (segment_counter < segments_.size() - 1 ||
           current_point == s.FirstPoint())) {
        const LineSegment2 shortcut =
            (current_point == s.FirstPoint())
                ? LineSegment2(segments_[segment_counter - 1].FirstPoint(),
                               s.SecondPoint())
                : LineSegment2(s.FirstPoint(),
                               segments_[segment_counter + 1].SecondPoint());
        current_signed_squared_distance *=
            (shortcut.Side(query)) ? sgn(current_signed_squared_distance)
                                   : -sgn(current_signed_squared_distance);

        CHECK(
            (current_signed_squared_distance >= 0.0 && shortcut.Side(query)) ||
            (current_signed_squared_distance <= 0.0 && !shortcut.Side(query)));
      }

      closest_signed_squared_distance = current_signed_squared_distance;
      closest_point = current_point;

      if (is_vertex) *is_vertex = is_segment_endpoint;
      segment_idx = segment_counter;
    }

    segment_counter++;
  }

  // Maybe set segment.
  if (segment) *segment = segments_[segment_idx];

  // Maybe set signed_squared_distance.
  if (signed_squared_distance)
    *signed_squared_distance = closest_signed_squared_distance;

  // Check if the closest point occurs at an endpoint for the polyline.
  if (is_endpoint) {
    auto is_same_point = [](const Point2& p1, const Point2& p2) {
      return (p1 - p2).squaredNorm() < constants::kSmallNumber;
    };  // is_same_point

    *is_endpoint =
        is_same_point(closest_point, segments_.front().FirstPoint()) ||
        is_same_point(closest_point, segments_.back().SecondPoint());
  }

  return closest_point;
}  // namespace ilqgames

}  // namespace ilqgames
