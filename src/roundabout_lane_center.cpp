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
// Compute a lane center entering and leaving a roundabout.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <vector>

namespace ilqgames {

PointList2 RoundaboutLaneCenter(float entrance_angle, float exit_angle,
                                float distance_from_roundabout) {
  // Radius of roundabout and lane half width.
  constexpr float kRoundaboutRadius = 12.0;  // m
  constexpr float kLaneHalfWidth = 2.5;      // m

  // Beginning of small 90 degree arc onto roundabout.
  const Point2 entry_arc_center(
      (kRoundaboutRadius + kLaneHalfWidth) * std::cos(entrance_angle),
      (kRoundaboutRadius + kLaneHalfWidth) * std::sin(entrance_angle));

  // Initial point in lane.
  const float first_entry_arc_point_angle = entrance_angle - M_PI_2;
  const Point2 first_point_in_entry_arc =
      entry_arc_center +
      kLaneHalfWidth * Point2(std::cos(first_entry_arc_point_angle),
                              std::sin(first_entry_arc_point_angle));

  PointList2 points = {
      first_point_in_entry_arc +
          distance_from_roundabout *
              Point2(std::cos(entrance_angle), std::sin(entrance_angle)),
      first_point_in_entry_arc};

  // Insert a short arc to take us into the roundabout.
  constexpr size_t kNumPointsInArc = 3;
  for (size_t ii = 1; ii <= kNumPointsInArc; ii++) {
    const float angle = first_entry_arc_point_angle -
                        M_PI_2 * static_cast<float>(ii) / kNumPointsInArc;
    points.push_back(entry_arc_center +
                     kLaneHalfWidth * Point2(std::cos(angle), std::sin(angle)));
  }

  // We should be back to the first point on the roundabout.
  const Point2 first_point_on_roundabout(
      kRoundaboutRadius * std::cos(entrance_angle),
      kRoundaboutRadius * std::sin(entrance_angle));
  CHECK_LT((points.back() - first_point_on_roundabout).norm(),
           constants::kSmallNumber);

  // Rest of the points in the roundabout.
  constexpr size_t kNumPointsInRoundabout = 10;
  for (size_t ii = 1; ii <= kNumPointsInRoundabout; ii++) {
    const float next_angle = entrance_angle + (exit_angle - entrance_angle) *
                                                  static_cast<float>(ii) /
                                                  kNumPointsInRoundabout;
    points.emplace_back(kRoundaboutRadius * std::cos(next_angle),
                        kRoundaboutRadius * std::sin(next_angle));
  }

  // Final point.
  constexpr float kFarAway = 1e4;  // m
  points.emplace_back(kFarAway * std::cos(exit_angle),
                      kFarAway * std::sin(exit_angle));

  return points;
}

}  // namespace ilqgames
