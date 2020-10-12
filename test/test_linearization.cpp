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
// Tests for linearization of dynamical systems.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/dynamics/air_3d.h>
#include <ilqgames/dynamics/concatenated_dynamical_system.h>
#include <ilqgames/dynamics/single_player_car_5d.h>
#include <ilqgames/dynamics/single_player_car_6d.h>
#include <ilqgames/dynamics/single_player_car_7d.h>
#include <ilqgames/dynamics/single_player_delayed_dubins_car.h>
#include <ilqgames/dynamics/single_player_dubins_car.h>
#include <ilqgames/dynamics/single_player_point_mass_2d.h>
#include <ilqgames/dynamics/single_player_unicycle_4d.h>
#include <ilqgames/dynamics/single_player_unicycle_5d.h>
#include <ilqgames/dynamics/two_player_unicycle_4d.h>
#include <ilqgames/utils/types.h>

#include <gtest/gtest.h>
#include <memory>
#include <random>

using namespace ilqgames;

namespace {
// Step size for forward differences.
static constexpr float kForwardStep = 1e-3;
static constexpr float kNumericalPrecision = 1e-2;

// Dubins car speed.
static constexpr float kDubinsSpeed = 1.0;

// Functions to compute numerical Jacobians.
void NumericalJacobian(const SinglePlayerDynamicalSystem& system, Time t,
                       const VectorXf& x, const VectorXf& u,
                       Eigen::Ref<MatrixXf> A, Eigen::Ref<MatrixXf> B) {
  // Check dimensions.
  EXPECT_EQ(system.XDim(), x.size());
  EXPECT_EQ(system.UDim(), u.size());
  EXPECT_EQ(A.rows(), system.XDim());
  EXPECT_EQ(A.cols(), system.XDim());
  EXPECT_EQ(B.rows(), system.XDim());
  EXPECT_EQ(B.cols(), system.UDim());

  // Compute each column of A by forward differences.
  const VectorXf xdot(system.Evaluate(t, x, u));
  for (Dimension ii = 0; ii < system.XDim(); ii++) {
    VectorXf x_forward(x);
    x_forward(ii) += kForwardStep;

    A.col(ii) += (system.Evaluate(t, x_forward, u) - xdot) * time::kTimeStep /
                 kForwardStep;
  }

  // Compute each column of B by forward differences.
  for (Dimension ii = 0; ii < system.UDim(); ii++) {
    VectorXf u_forward(u);
    u_forward(ii) += kForwardStep;

    B.col(ii) = (system.Evaluate(t, x, u_forward) - xdot) * time::kTimeStep /
                kForwardStep;
  }
}

LinearDynamicsApproximation NumericalJacobian(
    const MultiPlayerDynamicalSystem& system, Time t, const VectorXf& x,
    const std::vector<VectorXf>& us) {
  // Check dimensions.
  EXPECT_EQ(system.XDim(), x.size());
  EXPECT_EQ(system.NumPlayers(), us.size());
  for (size_t ii = 0; ii < system.NumPlayers(); ii++)
    EXPECT_EQ(system.UDim(ii), us[ii].size());

  // Create a new linearization.
  LinearDynamicsApproximation linearization(system);

  // Compute each column of A by forward differences.
  const VectorXf xdot(system.Evaluate(t, x, us));
  for (Dimension ii = 0; ii < system.XDim(); ii++) {
    VectorXf x_forward(x);
    x_forward(ii) += kForwardStep;

    linearization.A.col(ii) += (system.Evaluate(t, x_forward, us) - xdot) *
                               time::kTimeStep / kForwardStep;
  }

  // Compute each column of A by forward differences.
  std::vector<VectorXf> us_forward(us);
  for (size_t ii = 0; ii < system.NumPlayers(); ii++) {
    VectorXf& u_forward = us_forward[ii];
    MatrixXf& B = linearization.Bs[ii];

    for (Dimension jj = 0; jj < system.UDim(ii); jj++) {
      u_forward(jj) += kForwardStep;
      B.col(jj) = (system.Evaluate(t, x, us_forward) - xdot) * time::kTimeStep /
                  kForwardStep;
      u_forward(jj) -= kForwardStep;
    }
  }

  return linearization;
}

// Test that each system's linearization matches a numerical approximation.
void CheckLinearization(const SinglePlayerDynamicalSystem& system) {
  // Random number generator to make random timestamps.
  std::random_device rd;
  std::default_random_engine rng(rd());
  std::uniform_real_distribution<Time> time_distribution(0.0, 10.0);

  // Try a bunch of random points.
  constexpr size_t kNumRandomPoints = 10;
  for (size_t ii = 0; ii < kNumRandomPoints; ii++) {
    const VectorXf x(VectorXf::Random(system.XDim()));
    const VectorXf u(VectorXf::Random(system.UDim()));
    const Time t = time_distribution(rng);

    MatrixXf A_analytic(MatrixXf::Identity(system.XDim(), system.XDim()));
    MatrixXf B_analytic(MatrixXf::Zero(system.XDim(), system.UDim()));
    system.Linearize(t, x, u, A_analytic, B_analytic);

    MatrixXf A_numerical(MatrixXf::Identity(system.XDim(), system.XDim()));
    MatrixXf B_numerical(MatrixXf::Zero(system.XDim(), system.UDim()));
    NumericalJacobian(system, t, x, u, A_numerical, B_numerical);

    EXPECT_NEAR((A_analytic - A_numerical).cwiseAbs().maxCoeff(), 0.0,
                kNumericalPrecision);
    EXPECT_NEAR((B_analytic - B_numerical).cwiseAbs().maxCoeff(), 0.0,
                kNumericalPrecision);
  }
}

void CheckLinearization(const MultiPlayerDynamicalSystem& system) {
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

    const LinearDynamicsApproximation analytic = system.Linearize(t, x, us);
    const LinearDynamicsApproximation numerical =
        NumericalJacobian(system, t, x, us);

    EXPECT_NEAR((analytic.A - numerical.A).cwiseAbs().maxCoeff(), 0.0,
                kNumericalPrecision);
    for (size_t jj = 0; jj < system.NumPlayers(); jj++)
      EXPECT_NEAR((analytic.Bs[jj] - numerical.Bs[jj]).cwiseAbs().maxCoeff(),
                  0.0, kNumericalPrecision);
  }
}

}  // anonymous namespace

TEST(SinglePlayerDubinsTest, LinearizesCorrectly) {
  const SinglePlayerDubinsCar system(kDubinsSpeed);
  CheckLinearization(system);
}

TEST(SinglePlayerDelayedDubinsTest, LinearizesCorrectly) {
  const SinglePlayerDelayedDubinsCar system(kDubinsSpeed);
  CheckLinearization(system);
}

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

TEST(SinglePlayerCar6DTest, LinearizesCorrectly) {
  constexpr float kInterAxleLength = 4.0;  // m
  const SinglePlayerCar6D system(kInterAxleLength);
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

TEST(Air3DTest, LinearizesCorrectly) {
  constexpr float kSpeed = 3.0;  // m/s
  const Air3D system(kSpeed, kSpeed);
  CheckLinearization(system);
}

TEST(SinglePlayerPointMass2DTest, LinearizesCorrectly) {
  const SinglePlayerPointMass2D system;
  CheckLinearization(system);
}
