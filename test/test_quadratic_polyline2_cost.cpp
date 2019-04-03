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
// Tests for QuadraticCost.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/cost/quadratic_polyline2_cost.h>
#include <ilqgames/geometry/polyline2.h>
#include <ilqgames/utils/types.h>

#include <gtest/gtest.h>
#include <math.h>

using namespace ilqgames;

TEST(QuadraticPolyline2CostTest, QuadraticizeRandomPoints) {
  constexpr float kCostWeight = 5.0;
  constexpr Dimension kXIdx = 0;
  constexpr Dimension kYIdx = 1;

  // Set up polyline.
  const Point2 p1(0.0, -1.0);
  const Point2 p2(0.0, 1.0);
  const Point2 p3(2.0, 1.0);
  const Polyline2 polyline({p1, p2, p3});
  const QuadraticPolyline2Cost cost(kCostWeight, polyline, {kXIdx, kYIdx});

  // Hessian and gradient.
  constexpr Dimension kVectorDimension = 5;
  MatrixXf hess = MatrixXf::Zero(kVectorDimension, kVectorDimension);
  VectorXf grad = VectorXf::Zero(kVectorDimension);

  // Pick a bunch of random points and check Hessians and numerical gradients.
  constexpr size_t kNumRandomPoints = 10;
  for (size_t ii = 0; ii < kNumRandomPoints; ii++) {
    VectorXf input = VectorXf::Random(kVectorDimension);
    cost.Quadraticize(input, &hess, &grad);

    // Check Hessian.
    EXPECT_NEAR(hess(kXIdx, kXIdx), kCostWeight, constants::kSmallNumber);
    EXPECT_NEAR(hess(kYIdx, kYIdx), kCostWeight, constants::kSmallNumber);
    EXPECT_NEAR(hess.norm(), kCostWeight * std::sqrt(2.0),
                constants::kSmallNumber);

    // Compute numerical gradient and compare.
    const float value = cost.Evaluate(input);

    constexpr double kStepSize = 1e-3;
    input(kXIdx) += kStepSize;
    const float perturbed_value_x = cost.Evaluate(input);
    input(kXIdx) -= kStepSize;
    input(kYIdx) += kStepSize;
    const float perturbed_value_y = cost.Evaluate(input);

    const float deriv_x = (perturbed_value_x - value) / kStepSize;
    const float deriv_y = (perturbed_value_y - value) / kStepSize;

    EXPECT_NEAR(grad(kXIdx), deriv_x, 1e-2);
    EXPECT_NEAR(grad(kYIdx), deriv_y, 1e-2);
    EXPECT_NEAR(grad.norm(), std::hypot(deriv_x, deriv_y), 1e-2);

    // Reset hess/grad.
    hess(kXIdx, kXIdx) = hess(kYIdx, kYIdx) = 0.0;
    grad(kXIdx) = grad(kYIdx) = 0.0;
  }
}
