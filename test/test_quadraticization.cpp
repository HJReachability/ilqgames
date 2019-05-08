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
// Tests for quadraticization of costs.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/cost/curvature_cost.h>
#include <ilqgames/cost/nominal_path_length_cost.h>
#include <ilqgames/cost/proximity_cost.h>
#include <ilqgames/cost/quadratic_cost.h>
#include <ilqgames/cost/quadratic_polyline2_cost.h>
#include <ilqgames/cost/semiquadratic_cost.h>
#include <ilqgames/cost/semiquadratic_polyline2_cost.h>
#include <ilqgames/geometry/polyline2.h>
#include <ilqgames/utils/types.h>

#include <gtest/gtest.h>
#include <memory>
#include <random>

using namespace ilqgames;

namespace {
// Step size for forward differences.
static constexpr float kForwardStep = 1e-4;
static constexpr float kNumericalPrecision = 1e-2;

// Function to compute numerical gradient of a cost.
VectorXf NumericalGradient(const Cost& cost, Time t, const VectorXf& input) {
  VectorXf grad(input.size());

  // Evaluate original cost.
  const float original_cost = cost.Evaluate(t, input);

  // Forward differences.
  VectorXf query(input);
  for (size_t ii = 0; ii < input.size(); ii++) {
    query(ii) += kForwardStep;
    grad(ii) = (cost.Evaluate(t, query) - original_cost) / kForwardStep;
    query(ii) = input(ii);
  }

  return grad;
}

// Function to compute numerical Hessian of a cost.
MatrixXf NumericalHessian(const Cost& cost, Time t, const VectorXf& input) {
  MatrixXf hess(input.size(), input.size());

  // Original gradient.
  const VectorXf grad = NumericalGradient(cost, t, input);

  // Forward differences.
  VectorXf query(input);
  for (size_t ii = 0; ii < input.size(); ii++) {
    query(ii) += kForwardStep;
    hess.row(ii) = (NumericalGradient(cost, t, query) - grad) / kForwardStep;
    query(ii) = input(ii);
  }

  return hess;
}

// Test that each cost's gradient and Hessian match a numerical approximation.
void CheckQuadraticization(const Cost& cost, Dimension dim) {
  // Random number generator to make random timestamps.
  std::random_device rd;
  std::default_random_engine rng(rd());
  std::uniform_real_distribution<Time> time_distribution(0.0, 10.0);

  // Try a bunch of random points.
  constexpr size_t kNumRandomPoints = 10;
  for (size_t ii = 0; ii < kNumRandomPoints; ii++) {
    const VectorXf input(VectorXf::Random(dim));
    const Time t = time_distribution(rng);

    MatrixXf hess_analytic(MatrixXf::Zero(system.XDim(), system.XDim()));
    VectorXf grad_analytic(VectorXf::Zero(system.XDim(), system.UDim()));
    system.Linearize(t, kTimeStep, x, u, A_analytic, B_analytic);

    MatrixXf A_numerical(MatrixXf::Identity(system.XDim(), system.XDim()));
    MatrixXf B_numerical(MatrixXf::Zero(system.XDim(), system.UDim()));
    NumericalJacobian(system, t, kTimeStep, x, u, A_numerical, B_numerical);

    EXPECT_NEAR((A_analytic - A_numerical).cwiseAbs().maxCoeff(), 0.0,
                kNumericalPrecision);
    EXPECT_NEAR((B_analytic - B_numerical).cwiseAbs().maxCoeff(), 0.0,
                kNumericalPrecision);
  }
}

void CheckLinearization(const MultiPlayerDynamicalSystem& system) {
  constexpr Time kTimeStep = 0.1;

  // Random number generator to make random timestamps.
  std::random_device rd;
  std::default_random_engine rng(rd());
  std::uniform_real_distribution<Time> time_distribution(0.0, 10.0);

  // Try a bunch of random points.
  constexpr size_t kNumRandomPoints = 10;
  for (size_t ii = 0; ii < kNumRandomPoints; ii++) {
    const VectorXf x(VectorXf::Random(system.XDim()));
    const Time t = time_distribution(rng);

    std::vector<VectorXf> us(system.NumPlayers());
    for (size_t jj = 0; jj < system.NumPlayers(); jj++)
      us[jj] = VectorXf::Random(system.UDim(jj));

    const LinearDynamicsApproximation analytic =
        system.Linearize(t, kTimeStep, x, us);
    const LinearDynamicsApproximation numerical =
        NumericalJacobian(system, t, kTimeStep, x, us);

    EXPECT_NEAR((analytic.A - numerical.A).cwiseAbs().maxCoeff(), 0.0,
                kNumericalPrecision);
    for (size_t jj = 0; jj < system.NumPlayers(); jj++)
      EXPECT_NEAR((analytic.Bs[jj] - numerical.Bs[jj]).cwiseAbs().maxCoeff(),
                  0.0, kNumericalPrecision);
  }
}

}  // anonymous namespace

TEST(SinglePlayerUnicycle4DTest, LinearizesCorrectly) {
  const SinglePlayerUnicycle4D system;
  CheckLinearization(system);
}

TEST(SinglePlayerUnicycle5DTest, LinearizesCorrectly) {
  const SinglePlayerUnicycle5D system;
  CheckLinearization(system);
}

TEST(SinglePlayerCar5DTest, LinearizesCorrectly) {
  constexpr float kInterAxleLength = 4.0;  // m
  const SinglePlayerCar5D system(kInterAxleLength);
  CheckLinearization(system);
}

TEST(SinglePlayerCar7DTest, LinearizesCorrectly) {
  constexpr float kInterAxleLength = 4.0;  // m
  const SinglePlayerCar7D system(kInterAxleLength);
  CheckLinearization(system);
}

TEST(TwoPlayerUnicycle4DTest, LinearizesCorrectly) {
  const TwoPlayerUnicycle4D system;
  CheckLinearization(system);
}

TEST(ConcatenatedDynamicalSystemTest, LinearizesCorrectly) {
  constexpr float kInterAxleLength = 5.0;  // m
  const ConcatenatedDynamicalSystem system(
      {std::make_shared<SinglePlayerUnicycle4D>(),
       std::make_shared<SinglePlayerCar5D>(kInterAxleLength)});
  CheckLinearization(system);
}
