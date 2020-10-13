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
// Lane changing example in which the ego player (P1) sees clones of each other
// player, and costs it incurs due to other players are weighted averages of
// those from the clones.
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
#include <ilqgames/examples/lane_change_clone_example.h>
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

// Input contraints.
static constexpr float kOmegaMax = 1.5;  // rad/s
static constexpr float kAMax = 4.0;      // m/s

// Cost weights.
static constexpr float kOmegaCostWeight = 10.0;
static constexpr float kACostWeight = 5.0;
static constexpr float kNominalVCostWeight = 10.0;
static constexpr float kLaneCostWeight = 0.1;

// Probability distribution between P3a and P3b.
static constexpr float kP3aProbability = 0.25;
static constexpr float kP3bProbability = 1.0 - kP3aProbability;

// Nominal speed.
static constexpr float kP1NominalV = 2.0;    // m/s
static constexpr float kP2NominalV = 2.0;    // m/s
static constexpr float kP3aNominalV = 2.0;   // m/s
static constexpr float kP3bNominalV = 12.0;  // m/s

// Initial state.
static constexpr float kP1InitialX = 3.0;    // m
static constexpr float kP1InitialY = -20.0;  // m

static constexpr float kP2InitialX = 3.0;   // m
static constexpr float kP2InitialY = -5.0;  // m

static constexpr float kP3InitialX = -3.0;   // m
static constexpr float kP3InitialY = -40.0;  // m

static constexpr float kP1InitialTheta = M_PI_2;  // rad
static constexpr float kP1InitialV = 2.0;         // m/s

static constexpr float kP2InitialTheta = M_PI_2;  // rad
static constexpr float kP2InitialV = 2.0;         // m/s

static constexpr float kP3InitialTheta = M_PI_2;  // rad
static constexpr float kP3InitialV = 4.0;         // m/s

// State dimensions.
using P1 = SinglePlayerUnicycle4D;
using P2 = SinglePlayerUnicycle4D;
using P3 = SinglePlayerUnicycle4D;

static const Dimension kP1XIdx = P1::kPxIdx;
static const Dimension kP1YIdx = P1::kPyIdx;
static const Dimension kP1ThetaIdx = P1::kThetaIdx;
static const Dimension kP1VIdx = P1::kVIdx;

static const Dimension kP2XIdx = P1::kNumXDims + P2::kPxIdx;
static const Dimension kP2YIdx = P1::kNumXDims + P2::kPyIdx;
static const Dimension kP2ThetaIdx = P1::kNumXDims + P2::kThetaIdx;
static const Dimension kP2VIdx = P1::kNumXDims + P2::kVIdx;

static const Dimension kP3aXIdx = P1::kNumXDims + P2::kNumXDims + P3::kPxIdx;
static const Dimension kP3aYIdx = P1::kNumXDims + P2::kNumXDims + P3::kPyIdx;
static const Dimension kP3aThetaIdx =
    P1::kNumXDims + P2::kNumXDims + P3::kThetaIdx;
static const Dimension kP3aVIdx = P1::kNumXDims + P2::kNumXDims + P3::kVIdx;

static const Dimension kP3bXIdx =
    P1::kNumXDims + P2::kNumXDims + P3::kNumXDims + P3::kPxIdx;
static const Dimension kP3bYIdx =
    P1::kNumXDims + P2::kNumXDims + P3::kNumXDims + P3::kPyIdx;
static const Dimension kP3bThetaIdx =
    P1::kNumXDims + P2::kNumXDims + P3::kNumXDims + P3::kThetaIdx;
static const Dimension kP3bVIdx =
    P1::kNumXDims + P2::kNumXDims + P3::kNumXDims + P3::kVIdx;

}  // anonymous namespace

void LaneChangeCloneExample::ConstructDynamics() {
  // Create dynamics. In this case, we have three cars with decoupled unicycle
  // dynamics (they are only coupled through the cost structure of the game).
  // This is expressed in the ConcatenatedDynamicalSystem class. The third
  // player is "cloned" with two different nominal speeds and a probability
  // distribution / weighting between them.
  dynamics_.reset(new ConcatenatedDynamicalSystem(
      {std::make_shared<P1>(), std::make_shared<P2>(), std::make_shared<P3>(),
       std::make_shared<P3>()}));
}

void LaneChangeCloneExample::ConstructInitialState() {
  // Set up initial state. Initially, this is zero, but then we override
  // individual dimensions to match the desired initial conditions above.
  x0_ = VectorXf::Zero(dynamics_->XDim());
  x0_(kP1XIdx) = kP1InitialX;
  x0_(kP1YIdx) = kP1InitialY;
  x0_(kP1ThetaIdx) = kP1InitialTheta;
  x0_(kP1VIdx) = kP1InitialV;
  x0_(kP2XIdx) = kP2InitialX;
  x0_(kP2YIdx) = kP2InitialY;
  x0_(kP2ThetaIdx) = kP2InitialTheta;
  x0_(kP2VIdx) = kP2InitialV;
  x0_(kP3aXIdx) = kP3InitialX;
  x0_(kP3aYIdx) = kP3InitialY;
  x0_(kP3aThetaIdx) = kP3InitialTheta;
  x0_(kP3aVIdx) = kP3InitialV;
  x0_(kP3bXIdx) = kP3InitialX;
  x0_(kP3bYIdx) = kP3InitialY;
  x0_(kP3bThetaIdx) = kP3InitialTheta;
  x0_(kP3bVIdx) = kP3InitialV;
}

void LaneChangeCloneExample::ConstructPlayerCosts() {
  // Set up costs for all players. These are containers for holding each
  // player's constituent cost functions and constraints that hold pointwise in
  // time and can apply to either state or control (for *any* player).
  // These costs can also build in regularization on the state or the control,
  // which essentially boils down to adding a scaled identity matrix to each's
  // Hessian.
  player_costs_.emplace_back("P1");
  player_costs_.emplace_back("P2");
  player_costs_.emplace_back("P3a");
  player_costs_.emplace_back("P3b");
  auto& p1_cost = player_costs_[0];
  auto& p2_cost = player_costs_[1];
  auto& p3a_cost = player_costs_[2];
  auto& p3b_cost = player_costs_[3];

  // Quadratic control costs.
  // NOTE: ego (P1) wants to be polite to everyone else.
  const auto p1_omega_cost = std::make_shared<QuadraticCost>(
      kOmegaCostWeight, P1::kOmegaIdx, 0.0, "OmegaCost");
  const auto p1_a_cost = std::make_shared<QuadraticCost>(
      kACostWeight, P1::kAIdx, 0.0, "AccelerationCost");
  p1_cost.AddControlCost(0, p1_omega_cost);
  p1_cost.AddControlCost(0, p1_a_cost);

  const auto p1p2_omega_cost = std::make_shared<QuadraticCost>(
      kOmegaCostWeight, P2::kOmegaIdx, 0.0, "P2OmegaCost");
  const auto p1p2_a_cost = std::make_shared<QuadraticCost>(
      kACostWeight, P2::kAIdx, 0.0, "P2AccelerationCost");
  p1_cost.AddControlCost(1, p1p2_omega_cost);
  p1_cost.AddControlCost(1, p1p2_a_cost);

  const auto p1p3a_omega_cost = std::make_shared<QuadraticCost>(
      kP3aProbability * kOmegaCostWeight, P3::kOmegaIdx, 0.0, "P3aOmegaCost");
  const auto p1p3a_a_cost = std::make_shared<QuadraticCost>(
      kP3aProbability * kACostWeight, P3::kAIdx, 0.0, "P3aAccelerationCost");
  p1_cost.AddControlCost(2, p1p3a_omega_cost);
  p1_cost.AddControlCost(2, p1p3a_a_cost);

  const auto p1p3b_omega_cost = std::make_shared<QuadraticCost>(
      kP3bProbability * kOmegaCostWeight, P3::kOmegaIdx, 0.0, "P3bOmegaCost");
  const auto p1p3b_a_cost = std::make_shared<QuadraticCost>(
      kP3bProbability * kACostWeight, P3::kAIdx, 0.0, "P3bAccelerationCost");
  p1_cost.AddControlCost(3, p1p3b_omega_cost);
  p1_cost.AddControlCost(3, p1p3b_a_cost);

  const auto p2_omega_cost = std::make_shared<QuadraticCost>(
      kOmegaCostWeight, P2::kOmegaIdx, 0.0, "OmegaCost");
  const auto p2_a_cost = std::make_shared<QuadraticCost>(
      kACostWeight, P2::kAIdx, 0.0, "AccelerationCost");
  p2_cost.AddControlCost(1, p2_omega_cost);
  p2_cost.AddControlCost(1, p2_a_cost);

  const auto p3_omega_cost = std::make_shared<QuadraticCost>(
      kOmegaCostWeight, P3::kOmegaIdx, 0.0, "OmegaCost");
  const auto p3_a_cost = std::make_shared<QuadraticCost>(
      kACostWeight, P3::kAIdx, 0.0, "AccelerationCost");
  p3a_cost.AddControlCost(2, p3_omega_cost);
  p3a_cost.AddControlCost(2, p3_a_cost);
  p3b_cost.AddControlCost(3, p3_omega_cost);
  p3b_cost.AddControlCost(3, p3_a_cost);

  // Constrain each control input to lie in an interval.
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
  p3a_cost.AddControlConstraint(2, p3_omega_max_constraint);
  p3a_cost.AddControlConstraint(2, p3_omega_min_constraint);
  p3a_cost.AddControlConstraint(2, p3_a_max_constraint);
  p3a_cost.AddControlConstraint(2, p3_a_min_constraint);
  p3b_cost.AddControlConstraint(3, p3_omega_max_constraint);
  p3b_cost.AddControlConstraint(3, p3_omega_min_constraint);
  p3b_cost.AddControlConstraint(3, p3_a_max_constraint);
  p3b_cost.AddControlConstraint(3, p3_a_min_constraint);

  // Encourage each player to go a given nominal speed.
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

  // Encourage each player to remain near the lane center.
  const Polyline2 right_lane(
      {Point2(kP2InitialX, -1000.0), Point2(kP2InitialX, 1000.0)});
  const Polyline2 left_lane(
      {Point2(kP3InitialX, -1000.0), Point2(kP3InitialX, 1000.0)});

  const std::shared_ptr<QuadraticPolyline2Cost> p1_lane_cost(
      new QuadraticPolyline2Cost(kLaneCostWeight, left_lane, {kP1XIdx, kP1YIdx},
                                 "LaneCenter"));
  p1_cost.AddStateCost(p1_lane_cost);

  const std::shared_ptr<QuadraticPolyline2Cost> p2_lane_cost(
      new QuadraticPolyline2Cost(kLaneCostWeight, right_lane,
                                 {kP2XIdx, kP2YIdx}, "LaneCenter"));
  p2_cost.AddStateCost(p2_lane_cost);

  const std::shared_ptr<QuadraticPolyline2Cost> p3a_lane_cost(
      new QuadraticPolyline2Cost(kLaneCostWeight, left_lane,
                                 {kP3aXIdx, kP3aYIdx}, "LaneCenter"));
  p3a_cost.AddStateCost(p3a_lane_cost);

  const std::shared_ptr<QuadraticPolyline2Cost> p3b_lane_cost(
      new QuadraticPolyline2Cost(kLaneCostWeight, left_lane,
                                 {kP3bXIdx, kP3bYIdx}, "LaneCenter"));
  p3b_cost.AddStateCost(p3b_lane_cost);

  // Constrain all cars to stay in their lane (except P1 to stay on the road).
  constexpr float kLaneHalfWidth = 4.0;  // m
  const std::shared_ptr<Polyline2SignedDistanceConstraint> p1_road_constraint_l(
      new Polyline2SignedDistanceConstraint(left_lane, {kP1XIdx, kP1YIdx},
                                            -kLaneHalfWidth, false,
                                            "Left Road Boundary"));
  const std::shared_ptr<Polyline2SignedDistanceConstraint> p1_road_constraint_r(
      new Polyline2SignedDistanceConstraint(right_lane, {kP1XIdx, kP1YIdx},
                                            kLaneHalfWidth, true,
                                            "Right Road Boundary"));
  p1_cost.AddStateConstraint(p1_road_constraint_l);
  p1_cost.AddStateConstraint(p1_road_constraint_r);

  const std::shared_ptr<Polyline2SignedDistanceConstraint> p2_lane_constraint_l(
      new Polyline2SignedDistanceConstraint(right_lane, {kP2XIdx, kP2YIdx},
                                            -kLaneHalfWidth, false,
                                            "Left Lane Boundary"));
  const std::shared_ptr<Polyline2SignedDistanceConstraint> p2_lane_constraint_r(
      new Polyline2SignedDistanceConstraint(right_lane, {kP2XIdx, kP2YIdx},
                                            kLaneHalfWidth, true,
                                            "Right Lane Boundary"));
  p2_cost.AddStateConstraint(p2_lane_constraint_l);
  p2_cost.AddStateConstraint(p2_lane_constraint_r);

  const std::shared_ptr<Polyline2SignedDistanceConstraint>
      p3a_lane_constraint_l(new Polyline2SignedDistanceConstraint(
          left_lane, {kP3aXIdx, kP3aYIdx}, -kLaneHalfWidth, false,
          "Left Lane Boundary"));
  const std::shared_ptr<Polyline2SignedDistanceConstraint>
      p3a_lane_constraint_r(new Polyline2SignedDistanceConstraint(
          left_lane, {kP3aXIdx, kP3aYIdx}, kLaneHalfWidth, true,
          "Right Lane Boundary"));
  p3a_cost.AddStateConstraint(p3a_lane_constraint_l);
  p3a_cost.AddStateConstraint(p3a_lane_constraint_r);

  const std::shared_ptr<Polyline2SignedDistanceConstraint>
      p3b_lane_constraint_l(new Polyline2SignedDistanceConstraint(
          left_lane, {kP3bXIdx, kP3bYIdx}, -kLaneHalfWidth, false,
          "Left Lane Boundary"));
  const std::shared_ptr<Polyline2SignedDistanceConstraint>
      p3b_lane_constraint_r(new Polyline2SignedDistanceConstraint(
          left_lane, {kP3bXIdx, kP3bYIdx}, kLaneHalfWidth, true,
          "Right Lane Boundary"));
  p3b_cost.AddStateConstraint(p3b_lane_constraint_l);
  p3b_cost.AddStateConstraint(p3b_lane_constraint_r);

  // Constrain proximity.
  // NOTE: P2 knows P3a is the real clone so there's no need to constrain
  // collision between P2 and P3b.
  constexpr float kMinProximity = 6.0;  // m
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

inline std::vector<float> LaneChangeCloneExample::Xs(const VectorXf& x) const {
  return {x(kP1XIdx), x(kP2XIdx), x(kP3aXIdx), x(kP3bXIdx)};
}

inline std::vector<float> LaneChangeCloneExample::Ys(const VectorXf& x) const {
  return {x(kP1YIdx), x(kP2YIdx), x(kP3aYIdx), x(kP3bYIdx)};
}

inline std::vector<float> LaneChangeCloneExample::Thetas(
    const VectorXf& x) const {
  return {x(kP1ThetaIdx), x(kP2ThetaIdx), x(kP3aThetaIdx), x(kP3bThetaIdx)};
}

}  // namespace ilqgames
