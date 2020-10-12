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
// Three player collision-avoidance example using approximate HJ reachability.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/constraint/single_dimension_constraint.h>
#include <ilqgames/cost/extreme_value_cost.h>
#include <ilqgames/cost/quadratic_cost.h>
#include <ilqgames/cost/signed_distance_cost.h>
#include <ilqgames/dynamics/concatenated_dynamical_system.h>
#include <ilqgames/dynamics/single_player_car_5d.h>
#include <ilqgames/examples/three_player_collision_avoidance_reachability_example.h>
#include <ilqgames/geometry/draw_shapes.h>
#include <ilqgames/geometry/polyline2.h>
#include <ilqgames/solver/ilq_solver.h>
#include <ilqgames/solver/problem.h>
#include <ilqgames/solver/solver_params.h>
#include <ilqgames/utils/types.h>

#include <math.h>
#include <memory>
#include <vector>

// Initial state command-line flags.
DEFINE_double(d0, 5.0, "Initial distance from the origin (m).");
DEFINE_double(v0, 5.0, "Initial speed (m/s).");

// Buffer for the signed distance cost.
DEFINE_double(buffer, 3.0, "Nominal signed distance cost (m).");

namespace ilqgames {

namespace {

// Input contraints and cost
static constexpr float kOmegaMax = 1.0;
static constexpr float kAMax = 0.1;
static constexpr float kControlCostWeight = 0.1;

// State dimensions.
using P1 = SinglePlayerCar5D;
using P2 = SinglePlayerCar5D;
using P3 = SinglePlayerCar5D;
static constexpr float kInterAxleDistance = 4.0;

static const Dimension kP1XIdx = P1::kPxIdx;
static const Dimension kP1YIdx = P1::kPyIdx;
static const Dimension kP1HeadingIdx = P1::kThetaIdx;
static const Dimension kP1VIdx = P1::kVIdx;

static const Dimension kP2XIdx = P1::kNumXDims + P2::kPxIdx;
static const Dimension kP2YIdx = P1::kNumXDims + P2::kPyIdx;
static const Dimension kP2HeadingIdx = P1::kNumXDims + P2::kThetaIdx;
static const Dimension kP2VIdx = P1::kNumXDims + P2::kVIdx;

static const Dimension kP3XIdx = P1::kNumXDims + P2::kNumXDims + P3::kPxIdx;
static const Dimension kP3YIdx = P1::kNumXDims + P2::kNumXDims + P3::kPyIdx;
static const Dimension kP3HeadingIdx =
    P1::kNumXDims + P2::kNumXDims + P3::kThetaIdx;
static const Dimension kP3VIdx = P1::kNumXDims + P2::kNumXDims + P3::kVIdx;

}  // anonymous namespace

void ThreePlayerCollisionAvoidanceReachabilityExample::ConstructDynamics() {
  dynamics_.reset(new ConcatenatedDynamicalSystem(
      {std::make_shared<P1>(kInterAxleDistance),
       std::make_shared<P2>(kInterAxleDistance),
       std::make_shared<P3>(kInterAxleDistance)}));
}

void ThreePlayerCollisionAvoidanceReachabilityExample::ConstructInitialState() {
  // Set up initial state.
  constexpr float kAnglePerturbation = 0.1;  // rad
  x0_ = VectorXf::Zero(dynamics_->XDim());
  x0_(kP1XIdx) = FLAGS_d0;
  x0_(kP1YIdx) = 0.0;
  x0_(kP1HeadingIdx) = -M_PI + kAnglePerturbation;
  x0_(kP1VIdx) = FLAGS_v0;
  x0_(kP2XIdx) = -0.5 * FLAGS_d0;
  x0_(kP2YIdx) = 0.5 * std::sqrt(3.0) * FLAGS_d0;
  x0_(kP2HeadingIdx) = -M_PI / 3.0 + kAnglePerturbation;
  x0_(kP2VIdx) = FLAGS_v0;
  x0_(kP3XIdx) = -0.5 * FLAGS_d0;
  x0_(kP3YIdx) = -0.5 * std::sqrt(3.0) * FLAGS_d0;
  x0_(kP3HeadingIdx) = M_PI / 3.0 + kAnglePerturbation;
  x0_(kP3VIdx) = FLAGS_v0;
}

void ThreePlayerCollisionAvoidanceReachabilityExample::ConstructPlayerCosts() {
  // Set up costs for all players.
  player_costs_.emplace_back("P1");
  player_costs_.emplace_back("P2");
  player_costs_.emplace_back("P3");
  auto& p1_cost = player_costs_[0];
  auto& p2_cost = player_costs_[1];
  auto& p3_cost = player_costs_[2];

  // Quadratic control costs.
  const auto control_cost = std::make_shared<QuadraticCost>(
      kControlCostWeight, -1, 0.0, "ControlCost");
  p1_cost.AddControlCost(0, control_cost);
  p2_cost.AddControlCost(1, control_cost);
  p3_cost.AddControlCost(2, control_cost);

  // Constrain control input.
  const auto p1_omega_max_constraint =
      std::make_shared<SingleDimensionConstraint>(
          P1::kOmegaIdx, kOmegaMax, true, "Omega Constraint (Max)");
  const auto p1_omega_min_constraint =
      std::make_shared<SingleDimensionConstraint>(
          P1::kOmegaIdx, -kOmegaMax, false, "Omega Constraint (Min)");
  const auto p1_a_max_constraint = std::make_shared<SingleDimensionConstraint>(
      P1::kAIdx, kAMax, true, "Acceleration Constraint (Max)");
  const auto p1_a_min_constraint = std::make_shared<SingleDimensionConstraint>(
      P1::kAIdx, -kAMax, false, "Acceleration Constraint (Min)");
  p1_cost.AddControlConstraint(0, p1_omega_max_constraint);
  p1_cost.AddControlConstraint(0, p1_omega_min_constraint);
  p1_cost.AddControlConstraint(0, p1_a_max_constraint);
  p1_cost.AddControlConstraint(0, p1_a_min_constraint);

  const auto p2_omega_max_constraint =
      std::make_shared<SingleDimensionConstraint>(
          P2::kOmegaIdx, kOmegaMax, true, "Omega Constraint (Max)");
  const auto p2_omega_min_constraint =
      std::make_shared<SingleDimensionConstraint>(
          P2::kOmegaIdx, -kOmegaMax, false, "Omega Constraint (Min)");
  const auto p2_a_max_constraint = std::make_shared<SingleDimensionConstraint>(
      P2::kAIdx, kAMax, true, "Acceleration Constraint (Max)");
  const auto p2_a_min_constraint = std::make_shared<SingleDimensionConstraint>(
      P2::kAIdx, -kAMax, false, "Acceleration Constraint (Min)");
  p2_cost.AddControlConstraint(1, p2_omega_max_constraint);
  p2_cost.AddControlConstraint(1, p2_omega_min_constraint);
  p2_cost.AddControlConstraint(1, p2_a_max_constraint);
  p2_cost.AddControlConstraint(1, p2_a_min_constraint);

  const auto p3_omega_max_constraint =
      std::make_shared<SingleDimensionConstraint>(
          P3::kOmegaIdx, kOmegaMax, true, "Omega Constraint (Max)");
  const auto p3_omega_min_constraint =
      std::make_shared<SingleDimensionConstraint>(
          P3::kOmegaIdx, -kOmegaMax, false, "Omega Constraint (Min)");
  const auto p3_a_max_constraint = std::make_shared<SingleDimensionConstraint>(
      P3::kAIdx, kAMax, true, "Acceleration Constraint (Max)");
  const auto p3_a_min_constraint = std::make_shared<SingleDimensionConstraint>(
      P3::kAIdx, -kAMax, false, "Acceleration Constraint (Min)");
  p3_cost.AddControlConstraint(2, p3_omega_max_constraint);
  p3_cost.AddControlConstraint(2, p3_omega_min_constraint);
  p3_cost.AddControlConstraint(2, p3_a_max_constraint);
  p3_cost.AddControlConstraint(2, p3_a_min_constraint);

  // Penalize proximity.
  const std::shared_ptr<SignedDistanceCost> p1_p2_collision_avoidance_cost(
      new SignedDistanceCost({kP1XIdx, kP1YIdx}, {kP2XIdx, kP2YIdx},
                             FLAGS_buffer));
  const std::shared_ptr<SignedDistanceCost> p1_p3_collision_avoidance_cost(
      new SignedDistanceCost({kP1XIdx, kP1YIdx}, {kP3XIdx, kP3YIdx},
                             FLAGS_buffer));
  const std::shared_ptr<SignedDistanceCost> p2_p3_collision_avoidance_cost(
      new SignedDistanceCost({kP2XIdx, kP2YIdx}, {kP3XIdx, kP3YIdx},
                             FLAGS_buffer));

  constexpr bool kTakeMin = false;
  const std::shared_ptr<ExtremeValueCost> p1_proximity_cost(
      new ExtremeValueCost(
          {p1_p2_collision_avoidance_cost, p1_p3_collision_avoidance_cost},
          kTakeMin, "Proximity"));
  const std::shared_ptr<ExtremeValueCost> p2_proximity_cost(
      new ExtremeValueCost(
          {p1_p2_collision_avoidance_cost, p2_p3_collision_avoidance_cost},
          kTakeMin, "Proximity"));
  const std::shared_ptr<ExtremeValueCost> p3_proximity_cost(
      new ExtremeValueCost(
          {p2_p3_collision_avoidance_cost, p1_p3_collision_avoidance_cost},
          kTakeMin, "Proximity"));
  p1_cost.AddStateCost(p1_proximity_cost);
  p2_cost.AddStateCost(p2_proximity_cost);
  p3_cost.AddStateCost(p3_proximity_cost);

  // Make sure costs are max-over-time.
  p1_cost.SetMaxOverTime();
  p2_cost.SetMaxOverTime();
  p3_cost.SetMaxOverTime();
}

inline std::vector<float> ThreePlayerCollisionAvoidanceReachabilityExample::Xs(
    const VectorXf& x) const {
  return {x(kP1XIdx), x(kP2XIdx), x(kP3XIdx)};
}

inline std::vector<float> ThreePlayerCollisionAvoidanceReachabilityExample::Ys(
    const VectorXf& x) const {
  return {x(kP1YIdx), x(kP2YIdx), x(kP3YIdx)};
}

inline std::vector<float>
ThreePlayerCollisionAvoidanceReachabilityExample::Thetas(
    const VectorXf& x) const {
  return {x(kP1HeadingIdx), x(kP2HeadingIdx), x(kP3HeadingIdx)};
}

}  // namespace ilqgames
