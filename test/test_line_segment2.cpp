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
// Tests for LineSegment2.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/geometry/line_segment2.h>
#include <ilqgames/utils/types.h>

#include <gtest/gtest.h>
#include <math.h>

using namespace ilqgames;

// Check that we error out on construction if the line segment is degenerate.
TEST(LineSegment2Test, DiesIfDegenerate) {
  ASSERT_DEATH(LineSegment2(Point2::Zero(), Point2::Zero()), "Check failed");
}

// Check that we find the correct closest point.
TEST(LineSegment2Test, ClosestPointWorks) {
  const Point2 lower = Point2(0.0, -1.0);
  const Point2 upper = Point2(0.0, 1.0);
  const LineSegment2 segment(lower, upper);
  float signed_squared_distance;
  bool is_endpoint;

  // Pick points in the right half plane and check closest points/distances.
  Point2 query(1.0, -2.0);
  Point2 closest =
      segment.ClosestPoint(query, &is_endpoint, &signed_squared_distance);
  EXPECT_TRUE(is_endpoint);
  EXPECT_TRUE(closest.isApprox(lower));
  EXPECT_NEAR(signed_squared_distance, 2.0, constants::kSmallNumber);

  query << 1.0, 0.0;
  closest = segment.ClosestPoint(query, &is_endpoint, &signed_squared_distance);
  EXPECT_FALSE(is_endpoint);
  EXPECT_LT(closest.squaredNorm(), constants::kSmallNumber);
  EXPECT_NEAR(signed_squared_distance, 1.0, constants::kSmallNumber);

  query << 1.0, 2.0;
  closest = segment.ClosestPoint(query, &is_endpoint, &signed_squared_distance);
  EXPECT_TRUE(is_endpoint);
  EXPECT_TRUE(closest.isApprox(upper));
  EXPECT_NEAR(signed_squared_distance, 2.0, constants::kSmallNumber);

  // Pick points in the left half plane and check closest points/distances.
  query << -1.0, -2.0;
  closest = segment.ClosestPoint(query, &is_endpoint, &signed_squared_distance);
  EXPECT_TRUE(is_endpoint);
  EXPECT_TRUE(closest.isApprox(lower));
  EXPECT_NEAR(signed_squared_distance, -2.0, constants::kSmallNumber);

  query << -1.0, 0.0;
  closest = segment.ClosestPoint(query, &is_endpoint, &signed_squared_distance);
  EXPECT_FALSE(is_endpoint);
  EXPECT_LT(closest.squaredNorm(), constants::kSmallNumber);
  EXPECT_NEAR(signed_squared_distance, -1.0, constants::kSmallNumber);

  query << -1.0, 2.0;
  closest = segment.ClosestPoint(query, &is_endpoint, &signed_squared_distance);
  EXPECT_TRUE(is_endpoint);
  EXPECT_TRUE(closest.isApprox(upper));
  EXPECT_NEAR(signed_squared_distance, -2.0, constants::kSmallNumber);
}
