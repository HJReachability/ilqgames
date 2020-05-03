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

  const Point2 return_point = segments_[idx].FirstPoint() +
         remaining * segments_[idx].UnitDirection();
  if (is_endpoint) {
    if(idx == 0){
      if(return_point == segments_[idx].FirstPoint())
        *is_endpoint = true;
    }
    else if(idx == segments_.size()-1){
      if(return_point == segments_[idx].SecondPoint())
        *is_endpoint = true;
    }
    else
      *is_endpoint = false;
  }

  return return_point;
}

Point2 Polyline2::ClosestPoint(const Point2& query, bool* is_vertex,
                               LineSegment2* segment,
                               float* signed_squared_distance, bool* is_endpoint) const {
  // Walk along each line segment and remember which was closest.
  float closest_signed_squared_distance = constants::kInfinity;
  Point2 closest_point;

  float current_signed_squared_distance;
  int segment_ind = 0;
  int segment_counter = 0;
  for (const auto& s : segments_) {
    bool is_segment_endpoint;
    const Point2 current_point =
        s.ClosestPoint(query, &is_segment_endpoint, &current_signed_squared_distance);

    if (std::abs(current_signed_squared_distance) <
        std::abs(closest_signed_squared_distance)) {
      closest_signed_squared_distance = current_signed_squared_distance;
      closest_point = current_point;

      if (is_vertex) *is_vertex = is_segment_endpoint;
      if (segment) *segment = s;
      segment_ind = segment_counter;
    }

    segment_counter++; 
  }

  // Check if the closest point occurs at an endpoint for the polyline.
  if (is_endpoint) {
    // Check if the closest point is on the first or last segment of the polyline. 
    if (segment_ind == 0) {
      // Check if the closest point is also an endpoint for the adjacent line segment.
      bool check_vertex;
      float ssd;
      segments_[segment_ind + 1].ClosestPoint(query, &check_vertex, &ssd);
      if (!check_vertex) {
        // If the closest point is not an internal endpoint (vertex) then return true.
        *is_endpoint = true;
      }
    } else if (segment_ind == segments_.size()-1) {
      // Check if the closest point is also an endpoint for the adjacent line segment.
      bool check_vertex;
      float ssd;
      segments_[segment_ind - 1].ClosestPoint(query, &check_vertex, &ssd);
      if (!check_vertex) {
        // If the closest point is not an internal endpoint (vertex) then return true.
        *is_endpoint = true;
      }
    } else
     *is_endpoint = false; 
  } 


  // Maybe set signed_squared_distance.
  if (signed_squared_distance)
    *signed_squared_distance = closest_signed_squared_distance;

  return closest_point;
}

}  // namespace ilqgames
