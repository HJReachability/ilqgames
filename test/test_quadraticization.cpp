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

#include <ilqgames/constraint/affine_scalar_constraint.h>
#include <ilqgames/constraint/affine_vector_constraint.h>
#include <ilqgames/constraint/constraint.h>
#include <ilqgames/constraint/polyline2_signed_distance_constraint.h>
#include <ilqgames/constraint/proximity_constraint.h>
#include <ilqgames/constraint/single_dimension_constraint.h>
#include <ilqgames/cost/curvature_cost.h>
#include <ilqgames/cost/extreme_value_cost.h>
#include <ilqgames/cost/locally_convex_proximity_cost.h>
#include <ilqgames/cost/nominal_path_length_cost.h>
#include <ilqgames/cost/orientation_cost.h>
#include <ilqgames/cost/polyline2_signed_distance_cost.h>
#include <ilqgames/cost/proximity_cost.h>
#include <ilqgames/cost/quadratic_cost.h>
#include <ilqgames/cost/quadratic_difference_cost.h>
#include <ilqgames/cost/quadratic_norm_cost.h>
#include <ilqgames/cost/quadratic_polyline2_cost.h>
#include <ilqgames/cost/relative_distance_cost.h>
#include <ilqgames/cost/route_progress_cost.h>
#include <ilqgames/cost/semiquadratic_cost.h>
#include <ilqgames/cost/semiquadratic_norm_cost.h>
#include <ilqgames/cost/semiquadratic_polyline2_cost.h>
#include <ilqgames/cost/signed_distance_cost.h>
#include <ilqgames/cost/weighted_convex_proximity_cost.h>
#include <ilqgames/dynamics/concatenated_flat_system.h>
#include <ilqgames/dynamics/multi_player_flat_system.h>
#include <ilqgames/dynamics/single_player_flat_unicycle_4d.h>
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
static constexpr float kHessForwardStep = 1e-3;
static constexpr float kNumericalPrecision = 0.15;
static constexpr float kNumericalPrecisionFraction = 0.1;

// Function to compute numerical gradient of a cost. `eval` is a function to
// evaluate the cost.
template <typename E>
VectorXf NumericalGradient(const E& eval, Time t, const VectorXf& input) {
  VectorXf grad(input.size());

  // Central differences.
  VectorXf query(input);
  for (size_t ii = 0; ii < input.size(); ii++) {
    query(ii) += kGradForwardStep;
    const float hi = eval(t, query);

    query(ii) = input(ii) - kGradForwardStep;
    const float lo = eval(t, query);

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
void CheckQuadraticization(const Cost& cost, bool is_constraint) {
  // Random number generator to make random timestamps.
  std::default_random_engine rng(0);
  std::uniform_real_distribution<Time> time_distribution(0.0,
                                                         time::kTimeHorizon);
  std::bernoulli_distribution sign_distribution;
  std::uniform_real_distribution<float> entry_distribution(0.5, 5.0);

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

    // Custom method for evaluating the cost/constraint.
    auto eval = [&cost, &is_constraint](Time t, const VectorXf& input) {
      if (is_constraint) {
        const auto& constraint = *static_cast<const Constraint*>(&cost);
        return constraint.EvaluateAugmentedLagrangian(t, input);
      }

      return cost.Evaluate(t, input);
    };

    VectorXf grad_numerical = NumericalGradient(eval, t, input);

#if 1
    if ((hess_analytic - hess_numerical).lpNorm<Eigen::Infinity>() >=
            std::max(kNumericalPrecision,
                     kNumericalPrecisionFraction *
                         hess_analytic.cwiseAbs().maxCoeff()) ||
        (grad_analytic - grad_numerical).lpNorm<Eigen::Infinity>() >=
            std::max(kNumericalPrecision,
                     kNumericalPrecisionFraction *
                         grad_analytic.cwiseAbs().maxCoeff())) {
      std::cout << "input: " << input.transpose() << std::endl;
      std::cout << "numeric hess: \n" << hess_numerical << std::endl;
      std::cout << "analytic hess: \n" << hess_analytic << std::endl;
      std::cout << "numeric grad: \n" << grad_numerical << std::endl;
      std::cout << "analytic grad: \n" << grad_analytic << std::endl;
    }
#endif

    EXPECT_LT(
        (hess_analytic - hess_numerical).lpNorm<Eigen::Infinity>(),
        std::max(kNumericalPrecision, kNumericalPrecisionFraction *
                                          hess_analytic.cwiseAbs().maxCoeff()));
    EXPECT_LT(
        (grad_analytic - grad_numerical).lpNorm<Eigen::Infinity>(),
        std::max(kNumericalPrecision, kNumericalPrecisionFraction *
                                          grad_analytic.cwiseAbs().maxCoeff()));
  }
}

}  // namespace

TEST(QuadraticCostTest, QuadraticizesCorrectly) {
  QuadraticCost cost(kCostWeight, -1, 1.0);
  CheckQuadraticization(cost, false);
}

TEST(QuadraticDifferenceCostTest, QuadraticizesCorrectly) {
  QuadraticDifferenceCost cost(kCostWeight, {0, 1}, {1, 2});
  CheckQuadraticization(cost, false);
}

TEST(RelativeDistanceCostTest, QuadraticizesCorrectly) {
  RelativeDistanceCost cost(kCostWeight, {0, 1}, {1, 2});
  CheckQuadraticization(cost, false);
}

TEST(QuadraticNormCostTest, QuadraticizesCorrectly) {
  QuadraticNormCost cost(kCostWeight, {1, 2}, 1.0);
  CheckQuadraticization(cost, false);
}

TEST(SemiquadraticCostTest, QuadraticizesCorrectly) {
  SemiquadraticCost cost(kCostWeight, 0, 0.0, true);
  CheckQuadraticization(cost, false);
}

TEST(SemiquadraticNormCostTest, QuadraticizesCorrectly) {
  SemiquadraticNormCost cost(kCostWeight, {1, 2}, 1.0, true);
  CheckQuadraticization(cost, false);
}

TEST(QuadraticPolyline2CostTest, QuadraticizesCorrectly) {
  Polyline2 polyline({Point2(-2.0, -2.0), Point2(0.5, 1.0), Point2(2.0, 2.0)});
  QuadraticPolyline2Cost cost(kCostWeight, polyline, {0, 1});
  CheckQuadraticization(cost, false);
}

TEST(RouteProgressCostTest, QuadraticizesCorrectly) {
  Polyline2 polyline({Point2(-2.0, -2.0), Point2(0.5, 1.0), Point2(2.0, 2.0)});
  constexpr float kNominalSpeed = 0.1;
  RouteProgressCost cost(kCostWeight, kNominalSpeed, polyline, {0, 1});
  CheckQuadraticization(cost, false);
}

TEST(SemiquadraticPolyline2CostTest, QuadraticizesCorrectly) {
  Polyline2 polyline(
      {Point2(-200.0, -200.0), Point2(0.5, 1.0), Point2(200.0, 200.0)});
  SemiquadraticPolyline2Cost cost(kCostWeight, polyline, {0, 1}, 0.5, true);
  CheckQuadraticization(cost, false);
}

TEST(CurvatureCostTest, QuadraticizesCorrectly) {
  CurvatureCost cost(kCostWeight, 0, 1);
  CheckQuadraticization(cost, false);
}

TEST(NominalPathLengthCostTest, QuadraticizesCorrectly) {
  NominalPathLengthCost cost(kCostWeight, 0, 1.0);
  CheckQuadraticization(cost, false);
}

TEST(ProximityCostTest, QuadraticizesCorrectly) {
  ProximityCost cost(kCostWeight, {0, 1}, {2, 3}, 0.0);
  CheckQuadraticization(cost, false);
}

TEST(LocallyConvexProximityCostTest, QuadraticizesCorrectly) {
  LocallyConvexProximityCost cost(kCostWeight, {0, 1}, {2, 3}, 0.0);
  CheckQuadraticization(cost, false);
}

TEST(WeightedConvexProximityCostTest, QuadraticizesCorrectly) {
  WeightedConvexProximityCost cost(kCostWeight, {0, 1}, {2, 3}, 4, 5, 0.0);
  CheckQuadraticization(cost, false);
}

TEST(OrientationCostTest, QuadraticizesCorrectly) {
  OrientationCost cost(kCostWeight, 1, M_PI_2);
  CheckQuadraticization(cost, false);
}

TEST(Polyline2SignedDistanceCostTest, QuadraticizesCorrectly) {
  Polyline2 polyline({Point2(-2.0, -2.0), Point2(0.5, 1.0), Point2(2.0, 2.0)});
  Polyline2SignedDistanceCost cost(polyline, {0, 1});
  CheckQuadraticization(cost, false);
}

TEST(SignedDistanceCostTest, QuadraticizesCorrectly) {
  SignedDistanceCost cost({0, 1}, {2, 3}, 5.0);
  CheckQuadraticization(cost, false);
}

TEST(ExtremeValueCostTest, QuadraticizesCorrectly) {
  const std::shared_ptr<const SignedDistanceCost> cost1(
      new SignedDistanceCost({0, 1}, {2, 3}, 5.0));
  const std::shared_ptr<const QuadraticCost> cost2(
      new QuadraticCost(kCostWeight, -1, 1.0));
  ExtremeValueCost cost({cost1, cost2}, true);
  CheckQuadraticization(cost, false);
}

TEST(AffineScalarConstraintTest, QuadraticizesCorrectly) {
  AffineScalarConstraint constraint(
      VectorXf::LinSpaced(kInputDimension, -1.0, 1.0), 0.5, false);
  CheckQuadraticization(constraint, true);
}

TEST(AffineVectorConstraintTest, QuadraticizesCorrectly) {
  AffineVectorConstraint constraint(
      10.0 * MatrixXf::Random(kInputDimension, kInputDimension),
      VectorXf::Random(kInputDimension), false);
  CheckQuadraticization(constraint, true);
}

TEST(ProximityConstraintTest, QuadraticizesCorrectly) {
  ProximityConstraint constraint({0, 1}, {2, 3}, 0.7, false);
  CheckQuadraticization(constraint, true);
}

TEST(Polyline2SignedDistanceConstraintTest, QuadraticizesCorrectly) {
  Polyline2 polyline({Point2(-2.0, -2.0), Point2(0.5, 1.0), Point2(2.0, 2.0)});
  Polyline2SignedDistanceConstraint constraint(polyline, {0, 1}, 10.0, true);
  CheckQuadraticization(constraint, true);
}

TEST(SingleDimensionConstraintTest, QuadraticizesCorrectly) {
  SingleDimensionConstraint constraint(0, 1.0, true);
  CheckQuadraticization(constraint, true);
}
