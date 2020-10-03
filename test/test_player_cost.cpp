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
// Tests for PlayerCost.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/cost/player_cost.h>
#include <ilqgames/cost/quadratic_cost.h>
#include <ilqgames/utils/quadratic_cost_approximation.h>
#include <ilqgames/utils/types.h>

#include <gtest/gtest.h>
#include <math.h>
#include <memory>

using namespace ilqgames;

namespace {
// Constants.
static constexpr float kCostWeight = 1.0;
static constexpr size_t kNumPlayers = 2;
static constexpr Dimension kVectorDimension = 5;
}  // anonymous namespace

class PlayerCostTest : public ::testing::Test {
 protected:
  void SetUp() {
    // Add a couple costs.
    player_cost_.AddStateCost(std::make_shared<QuadraticCost>(kCostWeight, -1));
    player_cost_.AddControlCost(
        0, std::make_shared<QuadraticCost>(kCostWeight, -1));
    player_cost_.AddControlCost(
        1, std::make_shared<QuadraticCost>(kCostWeight, -1));

    // Choose a random state and controls.
    x_ = VectorXf::Random(kVectorDimension);
    for (size_t ii = 0; ii < kNumPlayers; ii++)
      us_.emplace_back(VectorXf::Random(kVectorDimension));
  }

  // PlayerCost, state, and controls for each player.
  PlayerCost player_cost_;
  VectorXf x_;
  std::vector<VectorXf> us_;
};  // class PlayerCostTest

// Test that we evaluate correctly.
TEST_F(PlayerCostTest, EvaluateWorks) {
  const float value = player_cost_.Evaluate(0.0, x_, us_);
  const float expected =
      0.5 * (x_.squaredNorm() +
             std::accumulate(us_.begin(), us_.end(), 0.0,
                             [](float total, const VectorXf& item) {
                               return total + item.squaredNorm();
                             }));
  EXPECT_NEAR(value, expected, constants::kSmallNumber);
}

// Check that we quadraticize correctly when dimension >= 0.
TEST_F(PlayerCostTest, QuadraticizeWorks) {
  const QuadraticCostApproximation quad =
      player_cost_.Quadraticize(0.0, x_, us_);

  // Check state Hessian is just kCostzaWeight on the diagonal.
  EXPECT_TRUE(quad.state.hess.diagonal().isApprox(
      VectorXf::Constant(kVectorDimension, kCostWeight),
      constants::kSmallNumber));
  EXPECT_NEAR(quad.state.hess.norm(),
              kCostWeight * std::sqrt(static_cast<float>(kVectorDimension)),
              constants::kSmallNumber);

  // Check state gradient.
  EXPECT_TRUE(
      quad.state.grad.isApprox(kCostWeight * x_, constants::kSmallNumber));

  // Check control Hessians.
  for (const auto& pair : quad.control) {
    const auto& R = pair.second.hess;
    EXPECT_TRUE(
        R.diagonal().isApprox(VectorXf::Constant(kVectorDimension, kCostWeight),
                              constants::kSmallNumber));
    EXPECT_NEAR(R.norm(),
                kCostWeight * std::sqrt(static_cast<float>(kVectorDimension)),
                constants::kSmallNumber);
  }
}
