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
// One player reachability example. Single player choosing control to minimize
// max distance to a target square.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/constraint/single_dimension_constraint.h>
#include <ilqgames/cost/polyline2_signed_distance_cost.h>
#include <ilqgames/cost/quadratic_cost.h>
#include <ilqgames/dynamics/concatenated_dynamical_system.h>
#include <ilqgames/dynamics/single_player_dubins_car.h>
#include <ilqgames/examples/one_player_reachability_example.h>
#include <ilqgames/geometry/draw_shapes.h>
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
static constexpr Time kTimeStep = 0.1;     // s
static constexpr Time kTimeHorizon = 2.0;  // s
static constexpr size_t kNumTimeSteps =
    static_cast<size_t>(kTimeHorizon / kTimeStep);

// Reach or avoid?
static constexpr bool kReach = false;

// Target radius.
static constexpr float kTargetRadius = 2.5;

// Input constraint.
static constexpr float kOmegaMax = 1.0;

// Initial state.
static constexpr float kP1InitialX = 2.0;          // m
static constexpr float kP1InitialY = 2.0;          // m
static constexpr float kP1InitialTheta = -M_PI;  // rad

static constexpr float kSpeed = 1.0;  // m/s

// Target position.
static constexpr float kP1TargetX = 0.0;
static constexpr float kP1TargetY = 0.0;

// State dimensions.
using P1 = SinglePlayerDubinsCar;

static const Dimension kP1XIdx = P1::kPxIdx;
static const Dimension kP1YIdx = P1::kPyIdx;
static const Dimension kP1ThetaIdx = P1::kThetaIdx;

// Control dimensions.
static const Dimension kP1OmegaIdx = 0;
}  // anonymous namespace

OnePlayerReachabilityExample::OnePlayerReachabilityExample(
    const SolverParams& params) {
  // Create dynamics.
  const std::shared_ptr<const ConcatenatedDynamicalSystem> dynamics(
      new ConcatenatedDynamicalSystem({std::make_shared<P1>(kSpeed)},
                                      kTimeStep));

  // Set up initial state.
  x0_ = VectorXf::Zero(dynamics->XDim());
  x0_(kP1XIdx) = kP1InitialX;
  x0_(kP1YIdx) = kP1InitialY;
  x0_(kP1ThetaIdx) = kP1InitialTheta;

  // Set up initial strategies and operating point.
  strategies_.reset(new std::vector<Strategy>());
  for (PlayerIndex ii = 0; ii < dynamics->NumPlayers(); ii++)
    strategies_->emplace_back(kNumTimeSteps, dynamics->XDim(),
                              dynamics->UDim(ii));

  operating_point_.reset(
      new OperatingPoint(kNumTimeSteps, dynamics->NumPlayers(), 0.0, dynamics));
  constexpr size_t kNumTimeStepsInitialTurn = 3;
  for (size_t kk = 0; kk < kNumTimeStepsInitialTurn; kk++)
    operating_point_->us[kk][0](0) = -0.5;


  // Set up costs for all players.
  PlayerCost p1_cost("P1");

  // Penalize and constrain control effort.
  const auto p1_omega_cost = std::make_shared<QuadraticCost>(
      params.control_cost_weight, kP1OmegaIdx, 0.0, "Steering");
  p1_cost.AddControlCost(0, p1_omega_cost);

  const auto p1_omega_max_constraint =
    std::make_shared<SingleDimensionConstraint>(kP1OmegaIdx, kOmegaMax, false,
                                                "Input Constraint (Max)");
  const auto p1_omega_min_constraint =
    std::make_shared<SingleDimensionConstraint>(kP1OmegaIdx, -kOmegaMax, true,
                                                "Input Constraint (Min)");
  p1_cost.AddControlConstraint(0, p1_omega_max_constraint);
  p1_cost.AddControlConstraint(0, p1_omega_min_constraint);

  // Target cost.
  const Polyline2 circle =
      DrawCircle(Point2(kP1TargetX, kP1TargetY), kTargetRadius, 10);
  const std::shared_ptr<Polyline2SignedDistanceCost> p1_target_cost(
      new Polyline2SignedDistanceCost(circle, {kP1XIdx, kP1YIdx}, kReach,
                                      "Target"));

  p1_cost.AddStateCost(p1_target_cost);

  // Make sure costs are exponentiated.
  CHECK_GT(params.exponential_constant, 0.0);
  p1_cost.SetExponentialConstant(params.exponential_constant);

  // Set up solver.
  solver_.reset(new ILQSolver(dynamics, {p1_cost}, kTimeHorizon, params));
}

inline std::vector<float> OnePlayerReachabilityExample::Xs(
    const VectorXf& x) const {
  return {x(kP1XIdx)};
}

inline std::vector<float> OnePlayerReachabilityExample::Ys(
    const VectorXf& x) const {
  return {x(kP1YIdx)};
}

inline std::vector<float> OnePlayerReachabilityExample::Thetas(
    const VectorXf& x) const {
  return {x(kP1ThetaIdx)};
}

}  // namespace ilqgames
