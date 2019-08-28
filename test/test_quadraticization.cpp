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
#include <ilqgames/cost/locally_convex_proximity_cost.h>
#include <ilqgames/cost/weighted_convex_proximity_cost.h>
#include <ilqgames/cost/nominal_path_length_cost.h>
#include <ilqgames/cost/proximity_barrier_cost.h>
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
// Cost weight and dimension.
static constexpr float kCostWeight = 1.0;
static constexpr Dimension kInputDimension = 10;

// Step size for forward differences.
static constexpr float kGradForwardStep = 1e-3;
static constexpr float kHessForwardStep = 2e-3;
static constexpr float kNumericalPrecision = 1e-1;

// Function to compute numerical gradient of a cost.
VectorXf NumericalGradient(const Cost& cost, Time t, const VectorXf& input) {
  VectorXf grad(input.size());

  // Central differences.
  VectorXf query(input);
  for (size_t ii = 0; ii < input.size(); ii++) {
    query(ii) += kGradForwardStep;
    const float hi = cost.Evaluate(t, query);

    query(ii) = input(ii) - kGradForwardStep;
    const float lo = cost.Evaluate(t, query);

    grad(ii) = 0.5 * (hi - lo) / kGradForwardStep;
    query(ii) = input(ii);
  }

  return grad;
}

// Function to compute numerical Hessian of a cost.
MatrixXf NumericalHessian(const Cost& cost, Time t, const VectorXf& input) {
  MatrixXf hess(input.size(), input.size());

  // Central differences on analytic gradients (otherwise things get too noisy).
  MatrixXf hess_analytic(input.size(), input.size());
  VectorXf query(input);
  for (size_t ii = 0; ii < input.size(); ii++) {
    VectorXf grad_analytic_hi = VectorXf::Zero(input.size());
    VectorXf grad_analytic_lo = VectorXf::Zero(input.size());

    query(ii) += kHessForwardStep;
    cost.Quadraticize(t, query, &hess_analytic, &grad_analytic_hi);

    query(ii) = input(ii) - kHessForwardStep;
    cost.Quadraticize(t, query, &hess_analytic, &grad_analytic_lo);

    hess.col(ii) =
        0.5 * (grad_analytic_hi - grad_analytic_lo) / kHessForwardStep;
    query(ii) = input(ii);
  }

  return hess;
}

// Test that each cost's gradient and Hessian match a numerical approximation.
void CheckQuadraticization(const Cost& cost) {
  // Random number generator to make random timestamps.
  std::default_random_engine rng(0);
  std::uniform_real_distribution<Time> time_distribution(0.0, 10.0);
  std::bernoulli_distribution sign_distribution;
  std::uniform_real_distribution<float> entry_distribution(0.25, 5.0);

  // Try a bunch of random points.
  constexpr size_t kNumRandomPoints = 20;
  for (size_t ii = 0; ii < kNumRandomPoints; ii++) {
    VectorXf input(kInputDimension);
    for (size_t jj = 0; jj < kInputDimension; jj++) {
      const float s = sign_distribution(rng);
      input(jj) = (1.0 - 2.0 * s) * entry_distribution(rng);
    }

    const Time t = time_distribution(rng);

    MatrixXf hess_analytic(MatrixXf::Zero(kInputDimension, kInputDimension));
    VectorXf grad_analytic(VectorXf::Zero(kInputDimension));
    cost.Quadraticize(t, input, &hess_analytic, &grad_analytic);

    MatrixXf hess_numerical = NumericalHessian(cost, t, input);
    VectorXf grad_numerical = NumericalGradient(cost, t, input);

#if 0
    if ((hess_analytic - hess_numerical).lpNorm<Eigen::Infinity>() >=
        kNumericalPrecision) {
      std::cout << "input: " << input.transpose() << std::endl;
      std::cout << "numeric hess: \n" << hess_numerical << std::endl;
      std::cout << "analytic hess: \n" << hess_analytic << std::endl;
      std::cout << "numeric grad: \n" << grad_numerical << std::endl;
      std::cout << "analytic grad: \n" << grad_analytic << std::endl;
    }
#endif

    EXPECT_LT((hess_analytic - hess_numerical).lpNorm<Eigen::Infinity>(),
              kNumericalPrecision);
    EXPECT_LT((grad_analytic - grad_numerical).lpNorm<Eigen::Infinity>(),
              kNumericalPrecision);
  }
}

}  // anonymous namespace

TEST(QuadraticCostTest, QuadraticizesCorrectly) {
  QuadraticCost cost(kCostWeight, -1, 1.0);
  CheckQuadraticization(cost);
}

TEST(SemiquadraticCostTest, QuadraticizesCorrectly) {
  SemiquadraticCost cost(kCostWeight, 0, 0.0, true);
  CheckQuadraticization(cost);
}

TEST(QuadraticPolyline2CostTest, QuadraticizesCorrectly) {
  Polyline2 polyline({Point2(-2.0, -2.0), Point2(0.5, 1.0), Point2(2.0, 2.0)});
  QuadraticPolyline2Cost cost(kCostWeight, polyline, {0, 1});
  CheckQuadraticization(cost);
}

TEST(SemiquadraticPolyline2CostTest, QuadraticizesCorrectly) {
  Polyline2 polyline(
      {Point2(-200.0, -200.0), Point2(0.5, 1.0), Point2(200.0, 200.0)});
  SemiquadraticPolyline2Cost cost(kCostWeight, polyline, {0, 1}, 0.5, true);
  CheckQuadraticization(cost);
}

TEST(CurvatureCostTest, QuadraticizesCorrectly) {
  CurvatureCost cost(kCostWeight, 0, 1);
  CheckQuadraticization(cost);
}

TEST(NominalPathLengthCostTest, QuadraticizesCorrectly) {
  NominalPathLengthCost cost(kCostWeight, 0, 1.0);
  CheckQuadraticization(cost);
}

TEST(ProximityCostTest, QuadraticizesCorrectly) {
  ProximityCost cost(kCostWeight, {0, 1}, {2, 3}, 0.0);
  CheckQuadraticization(cost);
}

TEST(ProximityBarrierCostTest, QuadraticizesCorrectly) {
  ProximityBarrierCost cost(kCostWeight, {0, 1}, {2, 3}, 0.0);
  CheckQuadraticization(cost);
}

TEST(LocallyConvexProximityCostTest, QuadraticizesCorrectly) {
  LocallyConvexProximityCost cost(kCostWeight, {0, 1}, {2, 3}, 0.0);
  CheckQuadraticization(cost);
}

TEST(WeightedConvexProximityCostTest, QuadraticizesCorrectly) {
  WeightedConvexProximityCost cost(kCostWeight, {0, 1}, {2, 3}, 4, 5, 0.0);
  CheckQuadraticization(cost);
}
