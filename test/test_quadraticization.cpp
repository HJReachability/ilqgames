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

#include <ilqgames/constraint/polyline2_signed_distance_constraint.h>
#include <ilqgames/constraint/proximity_constraint.h>
#include <ilqgames/constraint/single_dimension_constraint.h>
#include <ilqgames/cost/curvature_cost.h>
#include <ilqgames/cost/locally_convex_proximity_cost.h>
#include <ilqgames/cost/nominal_path_length_cost.h>
#include <ilqgames/cost/orientation_cost.h>
#include <ilqgames/cost/orientation_flat_cost.h>
#include <ilqgames/cost/proximity_cost.h>
#include <ilqgames/cost/quadratic_cost.h>
#include <ilqgames/cost/quadratic_norm_cost.h>
#include <ilqgames/cost/quadratic_polyline2_cost.h>
#include <ilqgames/cost/route_progress_cost.h>
#include <ilqgames/cost/semiquadratic_cost.h>
#include <ilqgames/cost/semiquadratic_norm_cost.h>
#include <ilqgames/cost/semiquadratic_polyline2_cost.h>
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

// Exponential constant in costs.
static constexpr float kExponentialConstant = 5.0;

// Function to compute numerical gradient of a cost.
VectorXf NumericalGradient(const Cost& cost, Time t, const VectorXf& input) {
  VectorXf grad(input.size());

  // Central differences.
  VectorXf query(input);
  for (size_t ii = 0; ii < input.size(); ii++) {
    query(ii) += kGradForwardStep;
    const float hi = (cost.IsExponentiated())
                         ? cost.EvaluateExponential(t, query)
                         : cost.Evaluate(t, query);

    query(ii) = input(ii) - kGradForwardStep;
    const float lo = (cost.IsExponentiated())
                         ? cost.EvaluateExponential(t, query)
                         : cost.Evaluate(t, query);

    grad(ii) = 0.5 * (hi - lo) / kGradForwardStep;
    query(ii) = input(ii);
  }

  return grad;
}

// VectorXf NumericalStateGradient(const GeneralizedControlCost& cost, Time t,
//                                 const VectorXf& xi, const VectorXf& v) {
//   VectorXf grad(xi.size());

//   // Central differences.
//   VectorXf query(xi);
//   for (size_t ii = 0; ii < xi.size(); ii++) {
//     query(ii) += kGradForwardStep;
//     const float hi = cost.Evaluate(t, query, v);

//     query(ii) = xi(ii) - kGradForwardStep;
//     const float lo = cost.Evaluate(t, query, v);

//     grad(ii) = 0.5 * (hi - lo) / kGradForwardStep;
//     query(ii) = xi(ii);
//   }

//   return grad;
// }

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

// MatrixXf NumericalStateHessian(const GeneralizedControlCost& cost, Time t,
//                                const VectorXf& xi, const VectorXf& v) {
//   MatrixXf hess(xi.size(), xi.size());

//   // Central differences on analytic gradients (otherwise things get too
//   noisy). MatrixXf hess_v(v.size(), v.size()); MatrixXf
//   hess_analytic(xi.size(), xi.size()); VectorXf query(xi); for (size_t ii =
//   0; ii < xi.size(); ii++) {
//     VectorXf grad_analytic_hi = VectorXf::Zero(xi.size());
//     VectorXf grad_analytic_lo = VectorXf::Zero(xi.size());

//     query(ii) += kHessForwardStep;
//     cost.Quadraticize(t, query, v, &hess_v, &hess_analytic,
//     &grad_analytic_hi);

//     query(ii) = xi(ii) - kHessForwardStep;
//     cost.Quadraticize(t, query, v, &hess_v, &hess_analytic,
//     &grad_analytic_lo);

//     hess.col(ii) =
//         0.5 * (grad_analytic_hi - grad_analytic_lo) / kHessForwardStep;
//     query(ii) = xi(ii);
//   }

//   return hess;
// }

// Test that each cost's gradient and Hessian match a numerical approximation.
void CheckQuadraticization(const Cost& cost) {
  // Random number generator to make random timestamps.
  std::default_random_engine rng(0);
  std::uniform_real_distribution<Time> time_distribution(0.0, 10.0);
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
    VectorXf grad_numerical = NumericalGradient(cost, t, input);

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

// void CheckQuadraticization(const GeneralizedControlCost& cost,
//                            const Cost& corresponding_cost,
//                            const ConcatenatedFlatSystem& dynamics,
//                            PlayerIndex player) {
//   // State and control dimension.
//   const Dimension xdim = dynamics.XDim();
//   const Dimension udim = dynamics.UDim(player);

//   // Random number generator to make random timestamps.
//   std::default_random_engine rng(0);
//   std::uniform_real_distribution<Time> time_distribution(0.0, 10.0);
//   std::bernoulli_distribution sign_distribution;
//   std::uniform_real_distribution<float> entry_distribution(0.25, 5.0);

//   // Try a bunch of random points.
//   constexpr size_t kNumRandomPoints = 20;
//   for (size_t ii = 0; ii < kNumRandomPoints; ii++) {
//     VectorXf xi(xdim);
//     for (size_t jj = 0; jj < xdim; jj++) {
//       const float s = sign_distribution(rng);
//       xi(jj) = (1.0 - 2.0 * s) * entry_distribution(rng);
//     }

//     VectorXf v(udim);
//     for (size_t jj = 0; jj < udim; jj++) {
//       v(jj) = entry_distribution(rng);
//     }

//     const Time t = time_distribution(rng);

//     // Compute all the analytic Hessians and gradient.
//     MatrixXf hess_v_analytic(MatrixXf::Zero(udim, udim));
//     MatrixXf hess_xi_analytic(MatrixXf::Zero(xdim, xdim));
//     VectorXf grad_xi_analytic(VectorXf::Zero(xdim));
//     cost.Quadraticize(t, xi, v, &hess_v_analytic, &hess_xi_analytic,
//                       &grad_xi_analytic);

//     // Numerical xi derivatives.
//     const MatrixXf hess_xi_numerical = NumericalStateHessian(cost, t, xi, v);
//     const VectorXf grad_xi_numerical = NumericalStateGradient(cost, t, xi,
//     v);

//     // Numerical v Hessian based on corresponding cost transformed by the
//     // inverse decoupling matrix.
//     const auto& subsystem = *dynamics.Subsystems()[player];
//     const VectorXf x = dynamics.FromLinearSystemState(xi, player);
//     const VectorXf u = subsystem.LinearizingControl(x, v);
//     MatrixXf hess_u(MatrixXf::Zero(udim, udim));
//     corresponding_cost.Quadraticize(t, u, &hess_u, nullptr);

//     const MatrixXf M_inv = subsystem.InverseDecouplingMatrix(x);
//     const MatrixXf hess_v_numerical = M_inv.transpose() * hess_u * M_inv;

// #if 0
//     if ((grad_xi_analytic - grad_xi_numerical).lpNorm<Eigen::Infinity>() >=
//         kNumericalPrecision) {
//       std::cout << "xi: " << xi.transpose() << std::endl;
//       std::cout << "v: " << v.transpose() << std::endl;
//       std::cout << "numeric hess: \n" << hess_xi_numerical << std::endl;
//       std::cout << "analytic hess: \n" << hess_xi_analytic << std::endl;
//       std::cout << "numeric grad: \n" << grad_xi_numerical << std::endl;
//       std::cout << "analytic grad: \n" << grad_xi_analytic << std::endl;
//     }
// #endif

//     EXPECT_LT((hess_v_analytic - hess_v_numerical).lpNorm<Eigen::Infinity>(),
//               kNumericalPrecision);
//     EXPECT_LT((hess_xi_analytic -
//     hess_xi_numerical).lpNorm<Eigen::Infinity>(),
//               kNumericalPrecision);
//     EXPECT_LT((grad_xi_analytic -
//     grad_xi_numerical).lpNorm<Eigen::Infinity>(),
//               kNumericalPrecision);
//   }
// }

}  // anonymous namespace

TEST(QuadraticCostTest, QuadraticizesCorrectly) {
  QuadraticCost cost(kCostWeight, -1, 1.0);
  CheckQuadraticization(cost);
}

TEST(QuadraticNormCostTest, QuadraticizesCorrectly) {
  QuadraticNormCost cost(kCostWeight, {1, 2}, 1.0);
  CheckQuadraticization(cost);
}

TEST(SemiquadraticCostTest, QuadraticizesCorrectly) {
  SemiquadraticCost cost(kCostWeight, 0, 0.0, true);
  CheckQuadraticization(cost);
}

TEST(SemiquadraticNormCostTest, QuadraticizesCorrectly) {
  SemiquadraticNormCost cost(kCostWeight, {1, 2}, 1.0, true);
  CheckQuadraticization(cost);
}

TEST(QuadraticPolyline2CostTest, QuadraticizesCorrectly) {
  Polyline2 polyline({Point2(-2.0, -2.0), Point2(0.5, 1.0), Point2(2.0, 2.0)});
  QuadraticPolyline2Cost cost(kCostWeight, polyline, {0, 1});
  CheckQuadraticization(cost);
}

TEST(RouteProgressCostTest, QuadraticizesCorrectly) {
  Polyline2 polyline({Point2(-2.0, -2.0), Point2(0.5, 1.0), Point2(2.0, 2.0)});
  constexpr float kNominalSpeed = 0.1;
  RouteProgressCost cost(kCostWeight, kNominalSpeed, polyline, {0, 1});
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

TEST(CurvatureCostTest, QuadraticizesExponentialCorrectly) {
  CurvatureCost cost(kCostWeight, 0, 1);
  cost.SetExponentialConstant(kExponentialConstant);
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

TEST(ProximityCostTest, QuadraticizesExponentialCorrectly) {
  ProximityCost cost(kCostWeight, {0, 1}, {2, 3}, 0.0);
  cost.SetExponentialConstant(kExponentialConstant);
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

TEST(OrientationFlatCostTest, QuadraticizesCorrectly) {
  OrientationFlatCost cost(kCostWeight, {1, 2}, 1.0);
  CheckQuadraticization(cost);
}

TEST(OrientationCostTest, QuadraticizesCorrectly) {
  OrientationCost cost(kCostWeight, 1, M_PI_2);
  CheckQuadraticization(cost);
}

TEST(ProximityConstraintTest, QuadraticizesCorrectly) {
  ProximityConstraint outside_constraint({0, 1}, {2, 3}, 0.0, false);
  CheckQuadraticization(outside_constraint);
}

TEST(Polyline2SignedDistanceConstraintTest, QuadraticizesCorrectly) {
  const Polyline2 polyline(
      {Point2(2.0, -2.0), Point2(-0.5, 1.0), Point2(2.0, 2.0)});
  Polyline2SignedDistanceConstraint left_constraint(polyline, {0, 1}, 10.0,
                                                    false);
  CheckQuadraticization(left_constraint);

  Polyline2SignedDistanceConstraint right_constraint(polyline, {0, 1}, -10.0,
                                                     true);
  CheckQuadraticization(right_constraint);
}

TEST(SingleDimensionConstraintTest, SingleDimensionCorrectly) {
  SingleDimensionConstraint left_constraint(0, 10.0, false);
  CheckQuadraticization(left_constraint);

  SingleDimensionConstraint right_constraint(0, -10.0, true);
  CheckQuadraticization(right_constraint);
}
