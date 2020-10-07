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
// Three player intersection example. Ordering is given by the following:
// (P1, P2, P3) = (Car 1, Car 2, Pedestrian).
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/constraint/polyline2_signed_distance_constraint.h>
#include <ilqgames/constraint/proximity_constraint.h>
#include <ilqgames/constraint/single_dimension_constraint.h>
#include <ilqgames/cost/curvature_cost.h>
#include <ilqgames/cost/final_time_cost.h>
#include <ilqgames/cost/locally_convex_proximity_cost.h>
#include <ilqgames/cost/nominal_path_length_cost.h>
#include <ilqgames/cost/proximity_cost.h>
#include <ilqgames/cost/quadratic_cost.h>
#include <ilqgames/cost/quadratic_polyline2_cost.h>
#include <ilqgames/cost/semiquadratic_cost.h>
#include <ilqgames/cost/semiquadratic_polyline2_cost.h>
#include <ilqgames/cost/weighted_convex_proximity_cost.h>
#include <ilqgames/dynamics/concatenated_dynamical_system.h>
#include <ilqgames/dynamics/single_player_car_5d.h>
#include <ilqgames/dynamics/single_player_car_6d.h>
#include <ilqgames/dynamics/single_player_unicycle_4d.h>
#include <ilqgames/examples/three_player_intersection_example.h>
#include <ilqgames/geometry/polyline2.h>
#include <ilqgames/solver/problem.h>
#include <ilqgames/solver/solver_params.h>
#include <ilqgames/utils/solver_log.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

#include <math.h>
#include <memory>
#include <vector>

namespace ilqgames {

namespace {

// Car inter-axle distance.
static constexpr float kInterAxleLength = 4.0;  // m

// Cost weights.
static constexpr float kStateRegularization = 1.0;
static constexpr float kControlRegularization = 5.0;

static constexpr float kOmegaCostWeight = 0.1;
static constexpr float kJerkCostWeight = 0.1;
static constexpr float kMaxOmega = 1.0;

static constexpr float kACostWeight = 0.1;
static constexpr float kNominalVCostWeight = 100.0;

static constexpr float kLaneCostWeight = 25.0;

static constexpr float kMinProximity = 6.0;
using ProxCost = ProximityCost;
static constexpr float kP1ProximityCostWeight = 10.0;
static constexpr float kP2ProximityCostWeight = 10.0;
static constexpr float kP3ProximityCostWeight = 10.0;

static constexpr bool kOrientedRight = true;
static constexpr bool kBarrierOrientedInside = false;

// Lane width.
static constexpr float kLaneHalfWidth = 2.5;  // m

// Nominal and max speed.
static constexpr float kP1MaxV = 12.0;  // m/s
static constexpr float kP2MaxV = 12.0;  // m/s
static constexpr float kP3MaxV = 2.0;   // m/s
static constexpr float kMinV = 1.0;     // m/s

static constexpr float kP1NominalV = 8.0;  // m/s
static constexpr float kP2NominalV = 5.0;  // m/s
static constexpr float kP3NominalV = 1.5;  // m/s

// Initial state.
static constexpr float kP1InitialX = -2.0;   // m
static constexpr float kP2InitialX = -10.0;  // m
static constexpr float kP3InitialX = -11.0;  // m

static constexpr float kP1InitialY = -30.0;  // m
static constexpr float kP2InitialY = 45.0;   // m
static constexpr float kP3InitialY = 16.0;   // m

static constexpr float kP1InitialHeading = M_PI_2;   // rad
static constexpr float kP2InitialHeading = -M_PI_2;  // rad
static constexpr float kP3InitialHeading = 0.0;      // rad

static constexpr float kP1InitialSpeed = 4.0;   // m/s
static constexpr float kP2InitialSpeed = 3.0;   // m/s
static constexpr float kP3InitialSpeed = 1.25;  // m/s

// State dimensions.
using P1 = SinglePlayerCar6D;
using P2 = SinglePlayerCar6D;
using P3 = SinglePlayerUnicycle4D;

static const Dimension kP1XIdx = P1::kPxIdx;
static const Dimension kP1YIdx = P1::kPyIdx;
static const Dimension kP1HeadingIdx = P1::kThetaIdx;
static const Dimension kP1PhiIdx = P1::kPhiIdx;
static const Dimension kP1VIdx = P1::kVIdx;
static const Dimension kP1AIdx = P1::kAIdx;

static const Dimension kP2XIdx = P1::kNumXDims + P2::kPxIdx;
static const Dimension kP2YIdx = P1::kNumXDims + P2::kPyIdx;
static const Dimension kP2HeadingIdx = P1::kNumXDims + P2::kThetaIdx;
static const Dimension kP2PhiIdx = P1::kNumXDims + P2::kPhiIdx;
static const Dimension kP2VIdx = P1::kNumXDims + P2::kVIdx;
static const Dimension kP2AIdx = P1::kNumXDims + P2::kAIdx;

static const Dimension kP3XIdx = P1::kNumXDims + P2::kNumXDims + P3::kPxIdx;
static const Dimension kP3YIdx = P1::kNumXDims + P2::kNumXDims + P3::kPyIdx;
static const Dimension kP3HeadingIdx =
    P1::kNumXDims + P2::kNumXDims + P3::kThetaIdx;
static const Dimension kP3VIdx = P1::kNumXDims + P2::kNumXDims + P3::kVIdx;

// Control dimensions.
static const Dimension kP1OmegaIdx = 0;
static const Dimension kP1JerkIdx = 1;
static const Dimension kP2OmegaIdx = 0;
static const Dimension kP2JerkIdx = 1;
static const Dimension kP3OmegaIdx = 0;
static const Dimension kP3AIdx = 1;
}  // anonymous namespace

void ThreePlayerIntersectionExample::ConstructDynamics() {
  dynamics_.reset(new ConcatenatedDynamicalSystem(
      {std::make_shared<P1>(kInterAxleLength),
       std::make_shared<P2>(kInterAxleLength), std::make_shared<P3>()}));
}

void ThreePlayerIntersectionExample::ConstructInitialState() {
  x0_ = VectorXf::Zero(dynamics_->XDim());
  x0_(kP1XIdx) = kP1InitialX;
  x0_(kP1YIdx) = kP1InitialY;
  x0_(kP1HeadingIdx) = kP1InitialHeading;
  x0_(kP1VIdx) = kP1InitialSpeed;
  x0_(kP2XIdx) = kP2InitialX;
  x0_(kP2YIdx) = kP2InitialY;
  x0_(kP2HeadingIdx) = kP2InitialHeading;
  x0_(kP2VIdx) = kP2InitialSpeed;
  x0_(kP3XIdx) = kP3InitialX;
  x0_(kP3YIdx) = kP3InitialY;
  x0_(kP3HeadingIdx) = kP3InitialHeading;
  x0_(kP3VIdx) = kP3InitialSpeed;
}

void ThreePlayerIntersectionExample::ConstructPlayerCosts() {
  // Set up costs for all players.
  player_costs_.emplace_back("P1", kStateRegularization,
                             kControlRegularization);
  player_costs_.emplace_back("P2", kStateRegularization,
                             kControlRegularization);
  player_costs_.emplace_back("P3", kStateRegularization,
                             kControlRegularization);
  auto& p1_cost = player_costs_[0];
  auto& p2_cost = player_costs_[1];
  auto& p3_cost = player_costs_[2];

  // Stay in lanes.
  const Polyline2 lane1(
      {Point2(kP1InitialX, -1000.0), Point2(kP1InitialX, 1000.0)});
  const Polyline2 lane2(
      {Point2(kP2InitialX, 1000.0), Point2(kP2InitialX, 18.0),
       Point2(kP2InitialX + 0.5, 15.0), Point2(kP2InitialX + 1.0, 14.0),
       Point2(kP2InitialX + 3.0, 12.5), Point2(kP2InitialX + 6.0, 12.0),
       Point2(1000.0, 12.0)});
  const Polyline2 lane3(
      {Point2(-1000.0, kP3InitialY), Point2(1000.0, kP3InitialY)});

  const std::shared_ptr<QuadraticPolyline2Cost> p1_lane_cost(
      new QuadraticPolyline2Cost(kLaneCostWeight, lane1, {kP1XIdx, kP1YIdx},
                                 "LaneCenter"));
  const std::shared_ptr<Polyline2SignedDistanceConstraint> p1_lane_r_constraint(
      new Polyline2SignedDistanceConstraint(lane1, {kP1XIdx, kP1YIdx},
                                            kLaneHalfWidth, !kOrientedRight,
                                            "LaneRightBoundary"));
  const std::shared_ptr<Polyline2SignedDistanceConstraint> p1_lane_l_constraint(
      new Polyline2SignedDistanceConstraint(lane1, {kP1XIdx, kP1YIdx},
                                            -kLaneHalfWidth, kOrientedRight,
                                            "LaneLeftBoundary"));
  p1_cost.AddStateCost(p1_lane_cost);
  // p1_cost.AddStateConstraint(p1_lane_r_constraint);
  // p1_cost.AddStateConstraint(p1_lane_l_constraint);

  const std::shared_ptr<QuadraticPolyline2Cost> p2_lane_cost(
      new QuadraticPolyline2Cost(kLaneCostWeight, lane2, {kP2XIdx, kP2YIdx},
                                 "LaneCenter"));
  const std::shared_ptr<Polyline2SignedDistanceConstraint> p2_lane_r_constraint(
      new Polyline2SignedDistanceConstraint(lane2, {kP2XIdx, kP2YIdx},
                                            kLaneHalfWidth, !kOrientedRight,
                                            "LaneRightBoundary"));
  const std::shared_ptr<Polyline2SignedDistanceConstraint> p2_lane_l_constraint(
      new Polyline2SignedDistanceConstraint(lane2, {kP2XIdx, kP2YIdx},
                                            -kLaneHalfWidth, kOrientedRight,
                                            "LaneLeftBoundary"));
  p2_cost.AddStateCost(p2_lane_cost);
  // p2_cost.AddStateConstraint(p2_lane_r_constraint);
  // p2_cost.AddStateConstraint(p2_lane_l_constraint);

  const std::shared_ptr<QuadraticPolyline2Cost> p3_lane_cost(
      new QuadraticPolyline2Cost(kLaneCostWeight, lane3, {kP3XIdx, kP3YIdx},
                                 "LaneCenter"));
  const std::shared_ptr<Polyline2SignedDistanceConstraint> p3_lane_r_constraint(
      new Polyline2SignedDistanceConstraint(lane3, {kP3XIdx, kP3YIdx},
                                            kLaneHalfWidth, !kOrientedRight,
                                            "LaneRightBoundary"));
  const std::shared_ptr<Polyline2SignedDistanceConstraint> p3_lane_l_constraint(
      new Polyline2SignedDistanceConstraint(lane3, {kP3XIdx, kP3YIdx},
                                            -kLaneHalfWidth, kOrientedRight,
                                            "LaneLeftBoundary"));
  p3_cost.AddStateCost(p3_lane_cost);
  // p3_cost.AddStateConstraint(p3_lane_r_constraint);
  // p3_cost.AddStateConstraint(p3_lane_l_constraint);

  // Max/min/nominal speed costs.
  const auto p1_min_v_constraint = std::make_shared<SingleDimensionConstraint>(
      kP1VIdx, kMinV, !kOrientedRight, "MinV");
  const auto p1_max_v_constraint = std::make_shared<SingleDimensionConstraint>(
      kP1VIdx, kP1MaxV, kOrientedRight, "MaxV");
  const auto p1_nominal_v_cost = std::make_shared<QuadraticCost>(
      kNominalVCostWeight, kP1VIdx, kP1NominalV, "NominalV");
  // p1_cost.AddStateConstraint(p1_min_v_constraint);
  // p1_cost.AddStateConstraint(p1_max_v_constraint);
  p1_cost.AddStateCost(p1_nominal_v_cost);

  const auto p2_min_v_constraint = std::make_shared<SingleDimensionConstraint>(
      kP2VIdx, kMinV, !kOrientedRight, "MinV");
  const auto p2_max_v_constraint = std::make_shared<SingleDimensionConstraint>(
      kP2VIdx, kP2MaxV, kOrientedRight, "MaxV");
  const auto p2_nominal_v_cost = std::make_shared<QuadraticCost>(
      kNominalVCostWeight, kP2VIdx, kP2NominalV, "NominalV");
  // p2_cost.AddStateConstraint(p2_min_v_constraint);
  // p2_cost.AddStateConstraint(p2_max_v_constraint);
  p2_cost.AddStateCost(p2_nominal_v_cost);

  const auto p3_min_v_constraint = std::make_shared<SingleDimensionConstraint>(
      kP3VIdx, kMinV, !kOrientedRight, "MinV");
  const auto p3_max_v_constraint = std::make_shared<SingleDimensionConstraint>(
      kP3VIdx, kP3MaxV, kOrientedRight, "MaxV");
  const auto p3_nominal_v_cost = std::make_shared<QuadraticCost>(
      kNominalVCostWeight, kP3VIdx, kP3NominalV, "NominalV");
  // p3_cost.AddStateConstraint(p3_min_v_constraint);
  // p3_cost.AddStateConstraint(p3_max_v_constraint);
  p3_cost.AddStateCost(p3_nominal_v_cost);

  // Penalize control effort.
  const auto p1_omega_max_constraint =
      std::make_shared<SingleDimensionConstraint>(
          kP1OmegaIdx, kMaxOmega, kOrientedRight, "SteeringMax");
  const auto p1_omega_min_constraint =
      std::make_shared<SingleDimensionConstraint>(
          kP1OmegaIdx, -kMaxOmega, !kOrientedRight, "SteeringMin");
  const auto p1_omega_cost = std::make_shared<QuadraticCost>(
      kOmegaCostWeight, kP1OmegaIdx, 0.0, "Steering");
  const auto p1_jerk_cost =
      std::make_shared<QuadraticCost>(kJerkCostWeight, kP1JerkIdx, 0.0, "Jerk");
  // p1_cost.AddControlConstraint(0, p1_omega_max_constraint);
  // p1_cost.AddControlConstraint(0, p1_omega_min_constraint);
  p1_cost.AddControlCost(0, p1_omega_cost);
  p1_cost.AddControlCost(0, p1_jerk_cost);

  const auto p2_omega_max_constraint =
      std::make_shared<SingleDimensionConstraint>(
          kP2OmegaIdx, kMaxOmega, !kOrientedRight, "SteeringMax");
  const auto p2_omega_min_constraint =
      std::make_shared<SingleDimensionConstraint>(
          kP2OmegaIdx, -kMaxOmega, kOrientedRight, "SteeringMin");
  const auto p2_omega_cost = std::make_shared<QuadraticCost>(
      kOmegaCostWeight, kP2OmegaIdx, 0.0, "Steering");
  const auto p2_jerk_cost =
      std::make_shared<QuadraticCost>(kJerkCostWeight, kP2JerkIdx, 0.0, "Jerk");
  // p2_cost.AddControlConstraint(1, p2_omega_max_constraint);
  // p2_cost.AddControlConstraint(1, p2_omega_min_constraint);
  p2_cost.AddControlCost(1, p2_omega_cost);
  p2_cost.AddControlCost(1, p2_jerk_cost);

  const auto p3_omega_max_constraint =
      std::make_shared<SingleDimensionConstraint>(
          kP3OmegaIdx, kMaxOmega, kOrientedRight, "SteeringMax");
  const auto p3_omega_min_constraint =
      std::make_shared<SingleDimensionConstraint>(
          kP3OmegaIdx, -kMaxOmega, !kOrientedRight, "SteeringMin");
  const auto p3_omega_cost = std::make_shared<QuadraticCost>(
      kOmegaCostWeight, kP3OmegaIdx, 0.0, "Steering");
  const auto p3_a_cost = std::make_shared<QuadraticCost>(kACostWeight, kP3AIdx,
                                                         0.0, "Acceleration");
  // p3_cost.AddControlConstraint(2, p3_omega_max_constraint);
  // p3_cost.AddControlConstraint(2, p3_omega_min_constraint);
  p3_cost.AddControlCost(2, p3_omega_cost);
  p3_cost.AddControlCost(2, p3_a_cost);

  // Pairwise proximity costs.
  // const std::shared_ptr<ProxCost> p1p2_proximity_cost(
  //     new ProxCost(kP1ProximityCostWeight, {kP1XIdx, kP1YIdx},
  //                  {kP2XIdx, kP2YIdx}, kMinProximity, "ProximityP2"));
  // const std::shared_ptr<ProxCost> p1p3_proximity_cost(
  //     new ProxCost(kP1ProximityCostWeight, {kP1XIdx, kP1YIdx},
  //                  {kP3XIdx, kP3YIdx}, kMinProximity, "ProximityP3"));
  // p1_cost.AddStateCost(p1p2_proximity_cost);
  // p1_cost.AddStateCost(p1p3_proximity_cost);

  // const std::shared_ptr<ProxCost> p2p1_proximity_cost(
  //     new ProxCost(kP2ProximityCostWeight, {kP2XIdx, kP2YIdx},
  //                  {kP1XIdx, kP1YIdx}, kMinProximity, "ProximityP1"));
  // const std::shared_ptr<ProxCost> p2p3_proximity_cost(
  //     new ProxCost(kP2ProximityCostWeight, {kP2XIdx, kP2YIdx},
  //                  {kP3XIdx, kP3YIdx}, kMinProximity, "ProximityP3"));
  // p2_cost.AddStateCost(p2p1_proximity_cost);
  // p2_cost.AddStateCost(p2p3_proximity_cost);

  // const std::shared_ptr<ProxCost> p3p1_proximity_cost(
  //     new ProxCost(kP3ProximityCostWeight, {kP3XIdx, kP3YIdx},
  //                  {kP1XIdx, kP1YIdx}, kMinProximity, "ProximityP1"));
  // const std::shared_ptr<ProxCost> p3p2_proximity_cost(
  //     new ProxCost(kP3ProximityCostWeight, {kP3XIdx, kP3YIdx},
  //                  {kP2XIdx, kP2YIdx}, kMinProximity, "ProximityP2"));
  // p3_cost.AddStateCost(p3p1_proximity_cost);
  // p3_cost.AddStateCost(p3p2_proximity_cost);

  // Collision-avoidance constraints.
  constexpr bool kKeepClose = true;
  const std::shared_ptr<ProximityConstraint> p1p2_proximity_constraint(
      new ProximityConstraint({kP1XIdx, kP1YIdx}, {kP2XIdx, kP2YIdx},
                              kMinProximity, !kKeepClose,
                              "ProximityConstraintP2"));
  const std::shared_ptr<ProximityConstraint> p1p3_proximity_constraint(
      new ProximityConstraint({kP1XIdx, kP1YIdx}, {kP3XIdx, kP3YIdx},
                              kMinProximity, !kKeepClose,
                              "ProximityConstraintP3"));
  p1_cost.AddStateConstraint(p1p2_proximity_constraint);
  p1_cost.AddStateConstraint(p1p3_proximity_constraint);

  const std::shared_ptr<ProximityConstraint> p2p1_proximity_constraint(
      new ProximityConstraint({kP2XIdx, kP2YIdx}, {kP1XIdx, kP1YIdx},
                              kMinProximity, !kKeepClose,
                              "ProximityConstraintP1"));
  const std::shared_ptr<ProximityConstraint> p2p3_proximity_constraint(
      new ProximityConstraint({kP2XIdx, kP2YIdx}, {kP3XIdx, kP3YIdx},
                              kMinProximity, !kKeepClose,
                              "ProximityConstraintP3"));
  p2_cost.AddStateConstraint(p2p1_proximity_constraint);
  p2_cost.AddStateConstraint(p2p3_proximity_constraint);

  const std::shared_ptr<ProximityConstraint> p3p1_proximity_constraint(
      new ProximityConstraint({kP3XIdx, kP3YIdx}, {kP1XIdx, kP1YIdx},
                              kMinProximity, !kKeepClose,
                              "ProximityConstraintP1"));
  const std::shared_ptr<ProximityConstraint> p3p2_proximity_constraint(
      new ProximityConstraint({kP3XIdx, kP3YIdx}, {kP2XIdx, kP2YIdx},
                              kMinProximity, !kKeepClose,
                              "ProximityConstraintP2"));
  p3_cost.AddStateConstraint(p3p1_proximity_constraint);
  p3_cost.AddStateConstraint(p3p2_proximity_constraint);

  // Collision-avoidance constraints.
  // const std::shared_ptr<ProximityConstraint> p1p2_proximity_constraint(
  //     new ProximityConstraint({kP1XIdx, kP1YIdx}, {kP2XIdx, kP2YIdx},
  //                          kMinProximity, kConstraintOrientedInside,
  //                          "ProximityConstraintP2"));
  // const std::shared_ptr<ProximityConstraint> p1p3_proximity_constraint(
  //     new ProximityConstraint({kP1XIdx, kP1YIdx}, {kP3XIdx, kP3YIdx},
  //                          kMinProximity, kConstraintOrientedInside,
  //                          "ProximityConstraintP3"));
  // p1_cost.AddStateConstraint(p1p2_proximity_constraint);
  // p1_cost.AddStateConstraint(p1p3_proximity_constraint);

  // const std::shared_ptr<ProximityConstraint> p2p1_proximity_constraint(
  //     new ProximityConstraint({kP2XIdx, kP2YIdx}, {kP1XIdx, kP1YIdx},
  //                          kMinProximity, kConstraintOrientedInside,
  //                          "ProximityConstraintP1"));
  // const std::shared_ptr<ProximityConstraint> p2p3_proximity_constraint(
  //     new ProximityConstraint({kP2XIdx, kP2YIdx}, {kP3XIdx, kP3YIdx},
  //                          kMinProximity, kConstraintOrientedInside,
  //                          "ProximityConstraintP3"));
  // p2_cost.AddStateConstraint(p2p1_proximity_constraint);
  // p2_cost.AddStateConstraint(p2p3_proximity_constraint);

  // const std::shared_ptr<ProximityConstraint> p3p1_proximity_constraint(
  //     new ProximityConstraint({kP3XIdx, kP3YIdx}, {kP1XIdx, kP1YIdx},
  //                          kMinProximity, kConstraintOrientedInside,
  //                          "ProximityConstraintP1"));
  // const std::shared_ptr<ProximityConstraint> p3p2_proximity_constraint(
  //     new ProximityConstraint({kP3XIdx, kP3YIdx}, {kP2XIdx, kP2YIdx},
  //                          kMinProximity, kConstraintOrientedInside,
  //                          "ProximityConstraintP2"));
  // p3_cost.AddStateConstraint(p3p1_proximity_constraint);
  // p3_cost.AddStateConstraint(p3p2_proximity_constraint);
}

inline std::vector<float> ThreePlayerIntersectionExample::Xs(
    const VectorXf& x) const {
  return {x(kP1XIdx), x(kP2XIdx), x(kP3XIdx)};
}

inline std::vector<float> ThreePlayerIntersectionExample::Ys(
    const VectorXf& x) const {
  return {x(kP1YIdx), x(kP2YIdx), x(kP3YIdx)};
}

inline std::vector<float> ThreePlayerIntersectionExample::Thetas(
    const VectorXf& x) const {
  return {x(kP1HeadingIdx), x(kP2HeadingIdx), x(kP3HeadingIdx)};
}

}  // namespace ilqgames
