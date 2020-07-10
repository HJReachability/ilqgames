/*
 * Copyright (c) 2020, The Regents of the University of California (Regents).
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
// Two player reachability example. Protagonist choosing control to minimize
// max distance (-ve) signed distance to a wall, and antagonist choosing
// disturbance to maximize max signed distance.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/constraint/single_dimension_constraint.h>
#include <ilqgames/cost/polyline2_signed_distance_cost.h>
#include <ilqgames/cost/quadratic_cost.h>
#include <ilqgames/dynamics/two_player_unicycle_4d.h>
#include <ilqgames/examples/two_player_reachability_example.h>
#include <ilqgames/geometry/polyline2.h>
#include <ilqgames/solver/ilq_solver.h>
#include <ilqgames/solver/problem.h>
#include <ilqgames/solver/solver_params.h>
#include <ilqgames/utils/types.h>

#include <math.h>
#include <memory>
#include <vector>

namespace ilqgames {

namespace {
// Time.
static constexpr Time kTimeStep = 0.1;      // s
static constexpr Time kTimeHorizon = 10.0;  // s
static constexpr size_t kNumTimeSteps =
    static_cast<size_t>(kTimeHorizon / kTimeStep);

// Reach or avoid?
static constexpr bool kAvoid = true;

// Input constraint.
static constexpr float kOmegaMax = 1.0;  // rad/s
static constexpr float kAMax = 1.0;      // m/s/s
static constexpr float kDMax = 0.5;      // m/s

// Initial state.
static constexpr float kInitialX = 0.0;      // m
static constexpr float kInitialY = -5.0;     // m
static constexpr float kInitialTheta = 0.5;  // rad
static constexpr float kInitialV = 1.0;      // m/s

// State dimensions.
using Dyn = TwoPlayerUnicycle4D;
}  // anonymous namespace

TwoPlayerReachabilityExample::TwoPlayerReachabilityExample(
    const SolverParams& params) {
  // Create dynamics.
  const auto dynamics = std::make_shared<const TwoPlayerUnicycle4D>(kTimeStep);

  // Set up initial state.
  x0_ = VectorXf::Zero(dynamics->XDim());
  x0_(Dyn::kPxIdx) = kInitialX;
  x0_(Dyn::kPyIdx) = kInitialY;
  x0_(Dyn::kThetaIdx) = kInitialTheta;
  x0_(Dyn::kVIdx) = kInitialV;

  // Set up initial strategies and operating point.
  strategies_.reset(new std::vector<Strategy>());
  for (PlayerIndex ii = 0; ii < dynamics->NumPlayers(); ii++)
    strategies_->emplace_back(kNumTimeSteps, dynamics->XDim(),
                              dynamics->UDim(ii));

  operating_point_.reset(
      new OperatingPoint(kNumTimeSteps, dynamics->NumPlayers(), 0.0, dynamics));
  constexpr size_t kNumTimeStepsInitialTurn = 0;
  for (size_t kk = 0; kk < kNumTimeStepsInitialTurn; kk++)
    operating_point_->us[kk][0](0) = -0.5;

  // Set up costs for all players.
  PlayerCost p1_cost("P1"), p2_cost("P2");

  // Penalize and constrain control effort.
  const auto p1_omega_cost = std::make_shared<QuadraticCost>(
      params.control_cost_weight, Dyn::kOmegaIdx, 0.0, "Steering");
  p1_cost.AddControlCost(0, p1_omega_cost);

  const auto p1_a_cost = std::make_shared<QuadraticCost>(
      params.control_cost_weight, Dyn::kAIdx, 0.0, "Acceleration");
  p1_cost.AddControlCost(0, p1_a_cost);

  const auto p2_dx_cost = std::make_shared<QuadraticCost>(
      params.control_cost_weight, Dyn::kDxIdx, 0.0, "Dx");
  p2_cost.AddControlCost(1, p2_dx_cost);

  const auto p2_dy_cost = std::make_shared<QuadraticCost>(
      params.control_cost_weight, Dyn::kDyIdx, 0.0, "Dy");
  p2_cost.AddControlCost(1, p2_dy_cost);

  const auto p1_omega_max_constraint =
      std::make_shared<SingleDimensionConstraint>(
          Dyn::kOmegaIdx, kOmegaMax, false, "Omega Constraint (Max)");
  const auto p1_omega_min_constraint =
      std::make_shared<SingleDimensionConstraint>(
          Dyn::kOmegaIdx, -kOmegaMax, true, "Omega Constraint (Min)");
  p1_cost.AddControlConstraint(0, p1_omega_max_constraint);
  p1_cost.AddControlConstraint(0, p1_omega_min_constraint);

  const auto p1_a_max_constraint = std::make_shared<SingleDimensionConstraint>(
      Dyn::kAIdx, kAMax, false, "Acceleration Constraint (Max)");
  const auto p1_a_min_constraint = std::make_shared<SingleDimensionConstraint>(
      Dyn::kAIdx, -kAMax, true, "Acceleration Constraint (Min)");
  p1_cost.AddControlConstraint(0, p1_a_max_constraint);
  p1_cost.AddControlConstraint(0, p1_a_min_constraint);

  const auto p2_dx_max_constraint = std::make_shared<SingleDimensionConstraint>(
      Dyn::kDxIdx, kDMax, false, "Dx Constraint (Max)");
  const auto p2_dx_min_constraint = std::make_shared<SingleDimensionConstraint>(
      Dyn::kDxIdx, -kDMax, true, "Dx Constraint (Min)");
  p2_cost.AddControlConstraint(1, p2_dx_max_constraint);
  p2_cost.AddControlConstraint(1, p2_dx_min_constraint);

  const auto p2_dy_max_constraint = std::make_shared<SingleDimensionConstraint>(
      Dyn::kDyIdx, kDMax, false, "Dy Constraint (Max)");
  const auto p2_dy_min_constraint = std::make_shared<SingleDimensionConstraint>(
      Dyn::kDyIdx, -kDMax, true, "Dy Constraint (Min)");
  p2_cost.AddControlConstraint(1, p2_dy_max_constraint);
  p2_cost.AddControlConstraint(1, p2_dy_min_constraint);

  // Target cost.
  const Polyline2 boundary({Point2(100.0, 0.0), Point2(-100.0, 0.0)});
  const std::shared_ptr<Polyline2SignedDistanceCost> p1_target_cost(
      new Polyline2SignedDistanceCost(boundary, {Dyn::kPxIdx, Dyn::kPyIdx},
                                      kAvoid, "Target"));
  const std::shared_ptr<Polyline2SignedDistanceCost> p2_target_cost(
      new Polyline2SignedDistanceCost(boundary, {Dyn::kPxIdx, Dyn::kPyIdx},
                                      kAvoid, "Target"));

  p1_cost.AddStateCost(p1_target_cost);
  p2_cost.AddStateCost(p2_target_cost);

  // Make sure costs are exponentiated.
  CHECK_GT(params.exponential_constant, 0.0);
  p1_cost.SetExponentialConstant(params.exponential_constant);
  p2_cost.SetExponentialConstant(params.exponential_constant);
  p2_cost.SetStateCostExponentialSign(-1.0);

  // Set up solver.
  solver_.reset(
      new ILQSolver(dynamics, {p1_cost, p2_cost}, kTimeHorizon, params));
}

inline std::vector<float> TwoPlayerReachabilityExample::Xs(
    const VectorXf& x) const {
  return {x(Dyn::kPxIdx)};
}

inline std::vector<float> TwoPlayerReachabilityExample::Ys(
    const VectorXf& x) const {
  return {x(Dyn::kPyIdx)};
}

inline std::vector<float> TwoPlayerReachabilityExample::Thetas(
    const VectorXf& x) const {
  return {x(Dyn::kThetaIdx)};
}

}  // namespace ilqgames
