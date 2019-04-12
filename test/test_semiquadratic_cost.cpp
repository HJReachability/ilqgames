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

#include <ilqgames/cost/quadratic_cost.h>
#include <ilqgames/cost/semiquadratic_cost.h>
#include <ilqgames/utils/types.h>

#include <gtest/gtest.h>
#include <math.h>

using namespace ilqgames;

// Check that the semiquadratic applies in the correct dimension.
TEST(SemiquadraticCostTest, EvaluatesInCorrectDimension) {
  constexpr float kCostWeight = 5.0;
  constexpr Dimension kDimension = 3;
  constexpr float kThreshold = 1.0;
  constexpr bool kOrientedRight = true;
  const SemiquadraticCost right_cost(kCostWeight, kDimension, kThreshold,
                                     kOrientedRight);
  const SemiquadraticCost left_cost(kCostWeight, kDimension, kThreshold,
                                    !kOrientedRight);

  // Try vectors of different lengths.
  ASSERT_DEATH(right_cost.Evaluate(VectorXf::Random(2)), "Check failed");
  ASSERT_DEATH(left_cost.Evaluate(VectorXf::Random(2)), "Check failed");

  VectorXf input(5);
  input << 0.5, 0.75, 1.5, 2.0, 2.5;
  float diff = input(kDimension) - kThreshold;
  EXPECT_NEAR(right_cost.Evaluate(input), kCostWeight * 0.5 * diff * diff,
              constants::kSmallNumber);
  EXPECT_NEAR(left_cost.Evaluate(input), 0.0, constants::kSmallNumber);

  input(kDimension) = -2.0;
  diff = input(kDimension) - kThreshold;
  EXPECT_NEAR(left_cost.Evaluate(input), kCostWeight * 0.5 * diff * diff,
              constants::kSmallNumber);
  EXPECT_NEAR(right_cost.Evaluate(input), 0.0, constants::kSmallNumber);
}

// Check that we quadraticize correctly when dimension >= 0.
TEST(SemiquadraticCostTest, QuadraticizeSingleDimension) {
  constexpr float kCostWeight = 5.0;
  constexpr Dimension kCostDimension = 3;
  constexpr Dimension kVectorDimension = 5;
  constexpr float kThreshold = 1.0;
  constexpr bool kOrientedRight = true;

  const SemiquadraticCost cost(kCostWeight, kCostDimension, kThreshold,
                               kOrientedRight);

  MatrixXf hess = MatrixXf::Zero(kVectorDimension, kVectorDimension);
  VectorXf grad = VectorXf::Zero(kVectorDimension);

  // Make sure we error out if the dimension is incorrect.
  ASSERT_DEATH(cost.Quadraticize(VectorXf::Random(2), &hess, &grad),
               "Check failed");

  // Compute gradient and Hessian at a couple points.
  VectorXf input(kVectorDimension);
  input << 0.5, 0.75, 1.5, 2.0, 2.5;
  float diff = input(kCostDimension) - kThreshold;
  const float deriv1 = kCostWeight * diff;

  cost.Quadraticize(input, &hess, &grad);
  EXPECT_NEAR(hess(kCostDimension, kCostDimension), kCostWeight,
              constants::kSmallNumber);
  EXPECT_NEAR(hess.norm(), std::abs(kCostWeight), constants::kSmallNumber);
  EXPECT_NEAR(grad(kCostDimension), deriv1, constants::kSmallNumber);
  EXPECT_NEAR(grad.norm(), std::abs(deriv1), constants::kSmallNumber);

  input(kCostDimension) = -2.0;
  const float deriv2 = 0.0;
  cost.Quadraticize(input, &hess, &grad);
  EXPECT_NEAR(hess(kCostDimension, kCostDimension), kCostWeight,
              constants::kSmallNumber);
  EXPECT_NEAR(hess.norm(), std::abs(kCostWeight), constants::kSmallNumber);
  EXPECT_NEAR(grad(kCostDimension), deriv1 + deriv2, constants::kSmallNumber);
  EXPECT_NEAR(grad.norm(), std::abs(deriv1 + deriv2), constants::kSmallNumber);
}
