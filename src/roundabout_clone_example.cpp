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
// Roundabout example in which the ego player (P1) sees clones of another
// player, and costs it incurs due to that other players are weighted averages
// of those from the clones. Each player is responsible for avoiding collision
// with the agent in the roundabout is is in conflict with.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/constraint/polyline2_signed_distance_constraint.h>
#include <ilqgames/constraint/proximity_constraint.h>
#include <ilqgames/constraint/single_dimension_constraint.h>
#include <ilqgames/cost/extreme_value_cost.h>
#include <ilqgames/cost/proximity_cost.h>
#include <ilqgames/cost/quadratic_cost.h>
#include <ilqgames/cost/quadratic_polyline2_cost.h>
#include <ilqgames/dynamics/concatenated_dynamical_system.h>
#include <ilqgames/dynamics/single_player_unicycle_4d.h>
#include <ilqgames/examples/roundabout_clone_example.h>
#include <ilqgames/examples/roundabout_lane_center.h>
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

// Cost weights.
static constexpr float kOmegaCostWeight = 50.0;
static constexpr float kACostWeight = 5.0;

static constexpr float kNominalVCostWeight = 1.0;
static constexpr float kLaneCostWeight = 1.0;
static constexpr float kMinProximity = 6.0;

// Lane width.
static constexpr float kLaneHalfWidth = 2.5;  // m

// Nominal and max speed.
static constexpr float kP1NominalV = 2.0;   // m/s
static constexpr float kP2NominalV = 5.0;   // m/s
static constexpr float kP3aNominalV = 5.0;  // m/s
static constexpr float kP3bNominalV = 6.0;  // m/s

static constexpr float kP1InitialSpeed = 5.0;  // m/s
static constexpr float kP2InitialSpeed = 5.0;  // m/s
static constexpr float kP3InitialSpeed = 5.0;  // m/s

// State dimensions.
static constexpr float kInterAxleDistance = 4.0;  // m
using Dyn = SinglePlayerUnicycle4D;

static const Dimension kP1XIdx = Dyn::kPxIdx;
static const Dimension kP1YIdx = Dyn::kPyIdx;
static const Dimension kP1ThetaIdx = Dyn::kThetaIdx;
static const Dimension kP1VIdx = Dyn::kVIdx;

static const Dimension kP2XIdx = Dyn::kNumXDims + Dyn::kPxIdx;
static const Dimension kP2YIdx = Dyn::kNumXDims + Dyn::kPyIdx;
static const Dimension kP2ThetaIdx = Dyn::kNumXDims + Dyn::kThetaIdx;
static const Dimension kP2VIdx = Dyn::kNumXDims + Dyn::kVIdx;

static const Dimension kP3aXIdx = Dyn::kNumXDims + Dyn::kNumXDims + Dyn::kPxIdx;
static const Dimension kP3aYIdx = Dyn::kNumXDims + Dyn::kNumXDims + Dyn::kPyIdx;
static const Dimension kP3aThetaIdx =
    Dyn::kNumXDims + Dyn::kNumXDims + Dyn::kThetaIdx;
static const Dimension kP3aVIdx = Dyn::kNumXDims + Dyn::kNumXDims + Dyn::kVIdx;

static const Dimension kP3bXIdx =
    Dyn::kNumXDims + Dyn::kNumXDims + Dyn::kNumXDims + Dyn::kPxIdx;
static const Dimension kP3bYIdx =
    Dyn::kNumXDims + Dyn::kNumXDims + Dyn::kNumXDims + Dyn::kPyIdx;
static const Dimension kP3bThetaIdx =
    Dyn::kNumXDims + Dyn::kNumXDims + Dyn::kNumXDims + Dyn::kThetaIdx;
static const Dimension kP3bVIdx =
    Dyn::kNumXDims + Dyn::kNumXDims + Dyn::kNumXDims + Dyn::kVIdx;

// Set up lanes for each player.
static constexpr float kInitialDistanceToRoundabout = 25.0;  // m
static constexpr float kAngleOffset = M_PI_2 * 0.5;
static constexpr float kP1WedgeSize = M_PI;
static constexpr float kP2WedgeSize = M_PI;
static constexpr float kP3aWedgeSize = M_PI;
static constexpr float kP3bWedgeSize = 0.5 * M_PI;
const std::vector<float> angles = {kAngleOffset, kAngleOffset + 0.5 * M_PI,
                                   kAngleOffset - 0.5 * M_PI,
                                   kAngleOffset - 0.5 * M_PI};
const Polyline2 lane1(RoundaboutLaneCenter(angles[0], angles[0] + kP1WedgeSize,
                                           kInitialDistanceToRoundabout));
const Polyline2 lane2(RoundaboutLaneCenter(angles[1], angles[1] + kP2WedgeSize,
                                           kInitialDistanceToRoundabout));
const Polyline2 lane3a(RoundaboutLaneCenter(angles[2],
                                            angles[2] + kP3aWedgeSize,
                                            kInitialDistanceToRoundabout));
const Polyline2 lane3b(RoundaboutLaneCenter(angles[3],
                                            angles[3] + kP3bWedgeSize,
                                            kInitialDistanceToRoundabout));

static const float kP1LanePosition = 0.0;
static const float kP2LanePosition = kInitialDistanceToRoundabout;
static const float kP3LanePosition = kInitialDistanceToRoundabout;

// Probability of P3 being "a" or "b".
static constexpr float kP3aProbability = 0.5;
static constexpr float kP3bProbability = 1.0 - kP3aProbability;

}  // anonymous namespace

void RoundaboutCloneExample::ConstructDynamics() {
  dynamics_.reset(new ConcatenatedDynamicalSystem(
      {std::make_shared<Dyn>(), std::make_shared<Dyn>(),
       std::make_shared<Dyn>(), std::make_shared<Dyn>()}));
}

void RoundaboutCloneExample::ConstructInitialState() {
  x0_ = VectorXf::Zero(dynamics_->XDim());

  const Point2 p1_pos = lane1.PointAt(kP1LanePosition, nullptr, nullptr,
                                      nullptr, &x0_(kP1ThetaIdx));
  const Point2 p2_pos = lane2.PointAt(kP2LanePosition, nullptr, nullptr,
                                      nullptr, &x0_(kP2ThetaIdx));
  const Point2 p3_pos = lane3a.PointAt(kP3LanePosition, nullptr, nullptr,
                                       nullptr, &x0_(kP3aThetaIdx));

  x0_(kP1XIdx) = p1_pos.x();
  x0_(kP1YIdx) = p1_pos.y();
  x0_(kP1VIdx) = kP1InitialSpeed;
  x0_(kP2XIdx) = p2_pos.x();
  x0_(kP2YIdx) = p2_pos.y();
  x0_(kP2VIdx) = kP2InitialSpeed;
  x0_(kP3aXIdx) = p3_pos.x();
  x0_(kP3aYIdx) = p3_pos.y();
  x0_(kP3aVIdx) = kP3InitialSpeed;
  x0_(kP3bXIdx) = p3_pos.x();
  x0_(kP3bYIdx) = p3_pos.y();
  x0_(kP3bThetaIdx) = x0_(kP3aThetaIdx);
  x0_(kP3bVIdx) = kP3InitialSpeed;
}

void RoundaboutCloneExample::ConstructInitialOperatingPoint() {
  // Initialize operating points to follow these lanes at the nominal speed.
  // InitializeAlongRoute(lane1, 0.0, kP1InitialSpeed, {kP1XIdx, kP1YIdx},
  //                      operating_point_.get());
  // InitializeAlongRoute(lane2, 0.0, kP2InitialSpeed, {kP2XIdx, kP2YIdx},
  //                      operating_point_.get());
  // InitializeAlongRoute(lane3, 0.0, kP3InitialSpeed, {kP3XIdx, kP3YIdx},
  //                      operating_point_.get());
  // InitializeAlongRoute(lane4, 0.0, kP4InitialSpeed, {kP4XIdx, kP4YIdx},
  //                      operating_point_.get());
  Problem::ConstructInitialOperatingPoint();
}

void RoundaboutCloneExample::ConstructPlayerCosts() {
  // Set up costs for all players.
  player_costs_.emplace_back("P1");
  player_costs_.emplace_back("P2");
  player_costs_.emplace_back("P3a");
  player_costs_.emplace_back("P3b");
  auto& p1_cost = player_costs_[0];
  auto& p2_cost = player_costs_[1];
  auto& p3a_cost = player_costs_[2];
  auto& p3b_cost = player_costs_[3];

  // Stay in lanes.
  const std::shared_ptr<QuadraticPolyline2Cost> p1_lane_cost(
      new QuadraticPolyline2Cost(kLaneCostWeight, lane1, {kP1XIdx, kP1YIdx},
                                 "Lane Center"));
  const std::shared_ptr<Polyline2SignedDistanceConstraint> p1_lane_constraint_l(
      new Polyline2SignedDistanceConstraint(lane1, {kP1XIdx, kP1YIdx},
                                            -kLaneHalfWidth, false,
                                            "Left Lane Boundary"));
  const std::shared_ptr<Polyline2SignedDistanceConstraint> p1_lane_constraint_r(
      new Polyline2SignedDistanceConstraint(lane1, {kP1XIdx, kP1YIdx},
                                            kLaneHalfWidth, true,
                                            "Right Lane Boundary"));
  p1_cost.AddStateCost(p1_lane_cost);
  p1_cost.AddStateConstraint(p1_lane_constraint_l);
  p1_cost.AddStateConstraint(p1_lane_constraint_r);

  const std::shared_ptr<QuadraticPolyline2Cost> p2_lane_cost(
      new QuadraticPolyline2Cost(kLaneCostWeight, lane2, {kP2XIdx, kP2YIdx},
                                 "Lane Center"));
  const std::shared_ptr<Polyline2SignedDistanceConstraint> p2_lane_constraint_l(
      new Polyline2SignedDistanceConstraint(lane2, {kP2XIdx, kP2YIdx},
                                            -kLaneHalfWidth, false,
                                            "Left Lane Boundary"));
  const std::shared_ptr<Polyline2SignedDistanceConstraint> p2_lane_constraint_r(
      new Polyline2SignedDistanceConstraint(lane2, {kP2XIdx, kP2YIdx},
                                            kLaneHalfWidth, true,
                                            "Right Lane Boundary"));
  p2_cost.AddStateCost(p2_lane_cost);
  p2_cost.AddStateConstraint(p2_lane_constraint_l);
  p2_cost.AddStateConstraint(p2_lane_constraint_r);

  const std::shared_ptr<QuadraticPolyline2Cost> p3a_lane_cost(
      new QuadraticPolyline2Cost(kLaneCostWeight, lane3a, {kP3aXIdx, kP3aYIdx},
                                 "Lane Center"));
  const std::shared_ptr<Polyline2SignedDistanceConstraint>
      p3a_lane_constraint_l(new Polyline2SignedDistanceConstraint(
          lane3a, {kP3aXIdx, kP3aYIdx}, -kLaneHalfWidth, false,
          "Left Lane Boundary"));
  const std::shared_ptr<Polyline2SignedDistanceConstraint>
      p3a_lane_constraint_r(new Polyline2SignedDistanceConstraint(
          lane3a, {kP3aXIdx, kP3aYIdx}, kLaneHalfWidth, true,
          "Right Lane Boundary"));
  p3a_cost.AddStateCost(p3a_lane_cost);
  p3a_cost.AddStateConstraint(p3a_lane_constraint_l);
  p3a_cost.AddStateConstraint(p3a_lane_constraint_r);

  const std::shared_ptr<QuadraticPolyline2Cost> p3b_lane_cost(
      new QuadraticPolyline2Cost(kLaneCostWeight, lane3b, {kP3bXIdx, kP3bYIdx},
                                 "Lane Center"));
  const std::shared_ptr<Polyline2SignedDistanceConstraint>
      p3b_lane_constraint_l(new Polyline2SignedDistanceConstraint(
          lane3b, {kP3bXIdx, kP3bYIdx}, -kLaneHalfWidth, false,
          "Left Lane Boundary"));
  const std::shared_ptr<Polyline2SignedDistanceConstraint>
      p3b_lane_constraint_r(new Polyline2SignedDistanceConstraint(
          lane3b, {kP3bXIdx, kP3bYIdx}, kLaneHalfWidth, true,
          "Right Lane Boundary"));
  p3b_cost.AddStateCost(p3b_lane_cost);
  p3b_cost.AddStateConstraint(p3b_lane_constraint_l);
  p3b_cost.AddStateConstraint(p3b_lane_constraint_r);

  // Max/min/nominal speed costs.
  const auto p1_nominal_v_cost = std::make_shared<QuadraticCost>(
      kNominalVCostWeight, kP1VIdx, kP1NominalV, "NominalV");
  p1_cost.AddStateCost(p1_nominal_v_cost);

  const auto p2_nominal_v_cost = std::make_shared<QuadraticCost>(
      kNominalVCostWeight, kP2VIdx, kP2NominalV, "NominalV");
  p2_cost.AddStateCost(p2_nominal_v_cost);

  const auto p3a_nominal_v_cost = std::make_shared<QuadraticCost>(
      kNominalVCostWeight, kP3aVIdx, kP3aNominalV, "NominalV");
  p3a_cost.AddStateCost(p3a_nominal_v_cost);

  const auto p3b_nominal_v_cost = std::make_shared<QuadraticCost>(
      kNominalVCostWeight, kP3bVIdx, kP3bNominalV, "NominalV");
  p3b_cost.AddStateCost(p3b_nominal_v_cost);

  // Penalize control effort.
  const auto omega_cost = std::make_shared<QuadraticCost>(
      kOmegaCostWeight, Dyn::kOmegaIdx, 0.0, "Steering");
  const auto a_cost =
      std::make_shared<QuadraticCost>(kACostWeight, Dyn::kAIdx, 0.0, "Accel");
  p1_cost.AddControlCost(0, omega_cost);
  p1_cost.AddControlCost(0, a_cost);
  p2_cost.AddControlCost(1, omega_cost);
  p2_cost.AddControlCost(1, a_cost);
  p3a_cost.AddControlCost(2, omega_cost);
  p3a_cost.AddControlCost(2, a_cost);
  p3b_cost.AddControlCost(3, omega_cost);
  p3b_cost.AddControlCost(3, a_cost);

  // P1 should also have a "politeness" cost for P3a/b, who is already in the
  // roundabout behind P1.
  const auto p3a_omega_politeness_cost = std::make_shared<QuadraticCost>(
      kP3aProbability * kOmegaCostWeight, Dyn::kOmegaIdx, 0.0,
      "Steering (Politeness P3a)");
  const auto p3a_a_politeness_cost = std::make_shared<QuadraticCost>(
      kP3aProbability * kACostWeight, Dyn::kAIdx, 0.0,
      "Accel (Politeness P3a)");
  const auto p3b_omega_politeness_cost = std::make_shared<QuadraticCost>(
      kP3bProbability * kOmegaCostWeight, Dyn::kOmegaIdx, 0.0,
      "Steering (Politeness P3b)");
  const auto p3b_a_politeness_cost = std::make_shared<QuadraticCost>(
      kP3bProbability * kACostWeight, Dyn::kAIdx, 0.0,
      "Accel (Politeness P3b)");
  p1_cost.AddControlCost(2, p3a_omega_politeness_cost);
  p1_cost.AddControlCost(2, p3a_a_politeness_cost);
  p1_cost.AddControlCost(3, p3b_omega_politeness_cost);
  p1_cost.AddControlCost(3, p3b_a_politeness_cost);

  // Proximity constraints. The agents in the rear bear collision-avoidance
  // constraints.
  const std::shared_ptr<ProximityConstraint> p1p2_proximity_constraint(
      new ProximityConstraint({kP1XIdx, kP1YIdx}, {kP2XIdx, kP2YIdx},
                              kMinProximity, false, "ProximityP1P2"));
  p1_cost.AddStateConstraint(p1p2_proximity_constraint);

  const std::shared_ptr<ProximityConstraint> p1p3a_proximity_constraint(
      new ProximityConstraint({kP1XIdx, kP1YIdx}, {kP3aXIdx, kP3aYIdx},
                              kMinProximity, false, "ProximityP1P3a"));
  p3a_cost.AddStateConstraint(p1p3a_proximity_constraint);

  const std::shared_ptr<ProximityConstraint> p1p3b_proximity_constraint(
      new ProximityConstraint({kP1XIdx, kP1YIdx}, {kP3bXIdx, kP3bYIdx},
                              kMinProximity, false, "ProximityP1P3b"));
  p3b_cost.AddStateConstraint(p1p3b_proximity_constraint);

  const std::shared_ptr<ProximityConstraint> p2p3a_proximity_constraint(
      new ProximityConstraint({kP2XIdx, kP2YIdx}, {kP3aXIdx, kP3aYIdx},
                              kMinProximity, false, "ProximityP2P3a"));
  p3a_cost.AddStateConstraint(p2p3a_proximity_constraint);

  const std::shared_ptr<ProximityConstraint> p2p3b_proximity_constraint(
      new ProximityConstraint({kP2XIdx, kP2YIdx}, {kP3bXIdx, kP3bYIdx},
                              kMinProximity, false, "ProximityP2P3b"));
  p3b_cost.AddStateConstraint(p2p3b_proximity_constraint);
}

inline std::vector<float> RoundaboutCloneExample::Xs(const VectorXf& x) const {
  return {x(kP1XIdx), x(kP2XIdx), x(kP3aXIdx), x(kP3bXIdx)};
}

inline std::vector<float> RoundaboutCloneExample::Ys(const VectorXf& x) const {
  return {x(kP1YIdx), x(kP2YIdx), x(kP3aYIdx), x(kP3bYIdx)};
}

inline std::vector<float> RoundaboutCloneExample::Thetas(
    const VectorXf& x) const {
  return {x(kP1ThetaIdx), x(kP2ThetaIdx), x(kP3aThetaIdx), x(kP3bThetaIdx)};
}

}  // namespace ilqgames
