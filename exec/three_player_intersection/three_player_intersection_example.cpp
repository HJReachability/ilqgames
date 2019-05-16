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

#include "three_player_intersection_example.h"
#include <ilqgames/cost/curvature_cost.h>
#include <ilqgames/cost/final_time_cost.h>
#include <ilqgames/cost/nominal_path_length_cost.h>
#include <ilqgames/cost/proximity_cost.h>
#include <ilqgames/cost/quadratic_cost.h>
#include <ilqgames/cost/quadratic_polyline2_cost.h>
#include <ilqgames/cost/semiquadratic_cost.h>
#include <ilqgames/cost/semiquadratic_polyline2_cost.h>
#include <ilqgames/dynamics/concatenated_dynamical_system.h>
#include <ilqgames/dynamics/single_player_car_7d.h>
#include <ilqgames/dynamics/single_player_unicycle_5d.h>
#include <ilqgames/geometry/polyline2.h>
#include <ilqgames/solver/ilq_solver.h>
#include <ilqgames/solver/linesearching_ilq_solver.h>
#include <ilqgames/solver/problem.h>
#include <ilqgames/utils/solver_log.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

#include <math.h>
#include <memory>
#include <vector>

namespace ilqgames {

namespace {
// Time.
static constexpr Time kTimeStep = 0.1;     // s
static constexpr Time kTimeHorizon = 5.0;  // s
static constexpr size_t kNumTimeSteps =
    static_cast<size_t>(kTimeHorizon / kTimeStep);

// Car inter-axle distance.
static constexpr float kInterAxleLength = 4.0;  // m

// Cost weights.
static constexpr float kACostWeight = 1.0;
static constexpr float kOmegaCostWeight = 50.0;
static constexpr float kCurvatureCostWeight = 0.0;

static constexpr float kMaxVCostWeight = 100.0;
static constexpr float kNominalVCostWeight = 1.0;

static constexpr float kSCostWeight = 0.0;
static constexpr float kGoalCostWeight = 100.0;

static constexpr float kLaneCostWeight = 20.0;
static constexpr float kLaneBoundaryCostWeight = 100.0;

static constexpr float kP0ProximityCostWeight = 100.0;
static constexpr float kP1ProximityCostWeight = 100.0;
static constexpr float kP2ProximityCostWeight = 10.0;

static constexpr bool kOrientedRight = true;

// Lane dimension.
static constexpr float kLaneHalfWidth = 2.0;  // m

// Goal points.
static constexpr float kP0GoalX = -6.0;  // m
static constexpr float kP0GoalY = 30.0;  // m

static constexpr float kP1GoalX = 20.0;  // m
static constexpr float kP1GoalY = 12.0;  // m

static constexpr float kP2GoalX = 5.0;   // m
static constexpr float kP2GoalY = 14.0;  // m

// Nominal and max speed.
static constexpr float kP0MaxV = 20.0;  // m/s
static constexpr float kP1MaxV = 20.0;  // m/s
static constexpr float kP2MaxV = 3.0;   // m/s
static constexpr float kMinV = 0.5;     // m/s

static constexpr float kP0NominalV = 10.0;  // m/s
static constexpr float kP1NominalV = 10.0;  // m/s
static constexpr float kP2NominalV = 1.0;   // m/s

// Initial state.
static constexpr float kP0InitialX = -5.0;   // m
static constexpr float kP1InitialX = -10.0;  // m
static constexpr float kP2InitialX = -12.0;  // m

static constexpr float kP0InitialY = -30.0;  // m
static constexpr float kP1InitialY = 30.0;   // m
static constexpr float kP2InitialY = 15.0;   // m

static constexpr float kP0InitialHeading = M_PI_2;   // rad
static constexpr float kP1InitialHeading = -M_PI_2;  // rad
static constexpr float kP2InitialHeading = 0.0;      // rad

static constexpr float kP0InitialSpeed = 8.0;   // m/s
static constexpr float kP1InitialSpeed = 8.0;   // m/s
static constexpr float kP2InitialSpeed = 0.75;  // m/s

// State dimensions.
using P0 = SinglePlayerCar7D;
using P1 = SinglePlayerCar7D;
using P2 = SinglePlayerUnicycle5D;

static constexpr Dimension kP0XIdx = P0::kPxIdx;
static constexpr Dimension kP0YIdx = P0::kPyIdx;
static constexpr Dimension kP0HeadingIdx = P0::kThetaIdx;
static constexpr Dimension kP0VIdx = P0::kVIdx;
static constexpr Dimension kP0KappaIdx = P0::kKappaIdx;
static constexpr Dimension kP0SIdx = P0::kSIdx;

static constexpr Dimension kP1XIdx = P0::kNumXDims + P1::kPxIdx;
static constexpr Dimension kP1YIdx = P0::kNumXDims + P1::kPyIdx;
static constexpr Dimension kP1HeadingIdx = P0::kNumXDims + P1::kThetaIdx;
static constexpr Dimension kP1VIdx = P0::kNumXDims + P1::kVIdx;
static constexpr Dimension kP1KappaIdx = P0::kNumXDims + P1::kKappaIdx;
static constexpr Dimension kP1SIdx = P0::kNumXDims + P1::kSIdx;

static constexpr Dimension kP2XIdx = P0::kNumXDims + P1::kNumXDims + P2::kPxIdx;
static constexpr Dimension kP2YIdx = P0::kNumXDims + P1::kNumXDims + P2::kPyIdx;
static constexpr Dimension kP2HeadingIdx =
    P0::kNumXDims + P1::kNumXDims + P2::kThetaIdx;
static constexpr Dimension kP2VIdx = P0::kNumXDims + P1::kNumXDims + P2::kVIdx;
static constexpr Dimension kP2SIdx = P0::kNumXDims + P1::kNumXDims + P2::kSIdx;

// Control dimensions.
static constexpr Dimension kP0OmegaIdx = 0;
static constexpr Dimension kP0AIdx = 1;
static constexpr Dimension kP1OmegaIdx = 0;
static constexpr Dimension kP1AIdx = 1;
static constexpr Dimension kP2OmegaIdx = 0;
static constexpr Dimension kP2AIdx = 1;
}  // anonymous namespace

ThreePlayerIntersectionExample::ThreePlayerIntersectionExample()
    : x_idxs_({kP0XIdx, kP1XIdx, kP2XIdx}),
      y_idxs_({kP0YIdx, kP1YIdx, kP2YIdx}),
      heading_idxs_({kP0HeadingIdx, kP1HeadingIdx, kP2HeadingIdx}) {
  // Create dynamics.
  const std::shared_ptr<ConcatenatedDynamicalSystem> dynamics(
      new ConcatenatedDynamicalSystem(
          {std::make_shared<SinglePlayerCar7D>(kInterAxleLength),
           std::make_shared<SinglePlayerCar7D>(kInterAxleLength),
           std::make_shared<SinglePlayerUnicycle5D>()}));

  // Set up initial state.
  x0_ = VectorXf::Zero(dynamics->XDim());
  x0_(kP0XIdx) = kP0InitialX;
  x0_(kP0YIdx) = kP0InitialY;
  x0_(kP0HeadingIdx) = kP0InitialHeading;
  x0_(kP0VIdx) = kP0InitialSpeed;
  x0_(kP1XIdx) = kP1InitialX;
  x0_(kP1YIdx) = kP1InitialY;
  x0_(kP1HeadingIdx) = kP1InitialHeading;
  x0_(kP1VIdx) = kP1InitialSpeed;
  x0_(kP2XIdx) = kP2InitialX;
  x0_(kP2YIdx) = kP2InitialY;
  x0_(kP2HeadingIdx) = kP2InitialHeading;
  x0_(kP2VIdx) = kP2InitialSpeed;

  // Set up initial strategies and operating point.
  strategies_.reset(new std::vector<Strategy>());
  for (PlayerIndex ii = 0; ii < dynamics->NumPlayers(); ii++)
    strategies_->emplace_back(kNumTimeSteps, dynamics->XDim(),
                              dynamics->UDim(ii));

  operating_point_.reset(
      new OperatingPoint(kNumTimeSteps, dynamics->NumPlayers(), dynamics));

  // Set up costs for all players.
  PlayerCost p0_cost, p1_cost, p2_cost;

  // Stay in lanes.
  const Polyline2 lane0(
      {Point2(kP0InitialX, -100.0), Point2(kP0InitialX, 100.0)});
  const Polyline2 lane1({Point2(kP1InitialX, 100.0), Point2(kP1InitialX, 18.0),
                         Point2(kP1InitialX + 0.5, 15.0),
                         Point2(kP1InitialX + 1.0, 14.0),
                         Point2(kP1InitialX + 3.0, 12.5),
                         Point2(kP1InitialX + 6.0, 12.0), Point2(100.0, 12.0)});
  const Polyline2 lane2(
      {Point2(-100.0, kP2InitialY), Point2(100.0, kP2InitialY)});

  const std::shared_ptr<QuadraticPolyline2Cost> p0_lane_cost(
      new QuadraticPolyline2Cost(kLaneCostWeight, lane0, {kP0XIdx, kP0YIdx},
                                 "LaneCenter"));
  const std::shared_ptr<SemiquadraticPolyline2Cost> p0_lane_r_cost(
      new SemiquadraticPolyline2Cost(kLaneBoundaryCostWeight, lane0,
                                     {kP0XIdx, kP0YIdx}, kLaneHalfWidth,
                                     kOrientedRight, "LaneRightBoundary"));
  const std::shared_ptr<SemiquadraticPolyline2Cost> p0_lane_l_cost(
      new SemiquadraticPolyline2Cost(kLaneBoundaryCostWeight, lane0,
                                     {kP0XIdx, kP0YIdx}, -kLaneHalfWidth,
                                     !kOrientedRight, "LaneLeftBoundary"));
  p0_cost.AddStateCost(p0_lane_cost);
  p0_cost.AddStateCost(p0_lane_r_cost);
  p0_cost.AddStateCost(p0_lane_l_cost);

  const std::shared_ptr<QuadraticPolyline2Cost> p1_lane_cost(
      new QuadraticPolyline2Cost(kLaneCostWeight, lane1, {kP1XIdx, kP1YIdx},
                                 "LaneCenter"));
  const std::shared_ptr<SemiquadraticPolyline2Cost> p1_lane_r_cost(
      new SemiquadraticPolyline2Cost(kLaneBoundaryCostWeight, lane1,
                                     {kP1XIdx, kP1YIdx}, kLaneHalfWidth,
                                     kOrientedRight, "LaneRightBoundary"));
  const std::shared_ptr<SemiquadraticPolyline2Cost> p1_lane_l_cost(
      new SemiquadraticPolyline2Cost(kLaneBoundaryCostWeight, lane1,
                                     {kP1XIdx, kP1YIdx}, -kLaneHalfWidth,
                                     !kOrientedRight, "LaneLeftBoundary"));
  p1_cost.AddStateCost(p1_lane_cost);
  p1_cost.AddStateCost(p1_lane_r_cost);
  p1_cost.AddStateCost(p1_lane_l_cost);

  const std::shared_ptr<QuadraticPolyline2Cost> p2_lane_cost(
      new QuadraticPolyline2Cost(kLaneCostWeight, lane2, {kP2XIdx, kP2YIdx},
                                 "LaneCenter"));
  const std::shared_ptr<SemiquadraticPolyline2Cost> p2_lane_r_cost(
      new SemiquadraticPolyline2Cost(kLaneBoundaryCostWeight, lane2,
                                     {kP2XIdx, kP2YIdx}, kLaneHalfWidth,
                                     kOrientedRight, "LaneRightBoundary"));
  const std::shared_ptr<SemiquadraticPolyline2Cost> p2_lane_l_cost(
      new SemiquadraticPolyline2Cost(kLaneBoundaryCostWeight, lane2,
                                     {kP2XIdx, kP2YIdx}, -kLaneHalfWidth,
                                     !kOrientedRight, "LaneLeftBoundary"));
  p2_cost.AddStateCost(p2_lane_cost);
  p2_cost.AddStateCost(p2_lane_r_cost);
  p2_cost.AddStateCost(p2_lane_l_cost);

  // Max/min/nominal speed costs.
  const auto p0_min_v_cost = std::make_shared<SemiquadraticCost>(
      kMaxVCostWeight, kP0VIdx, kMinV, !kOrientedRight, "MinV");
  const auto p0_max_v_cost = std::make_shared<SemiquadraticCost>(
      kMaxVCostWeight, kP0VIdx, kP0MaxV, kOrientedRight, "MaxV");
  const auto p0_nominal_v_cost = std::make_shared<QuadraticCost>(
      kNominalVCostWeight, kP0VIdx, kP0NominalV, "NominalV");
  p0_cost.AddStateCost(p0_min_v_cost);
  p0_cost.AddStateCost(p0_max_v_cost);
  p0_cost.AddStateCost(p0_nominal_v_cost);

  const auto p1_min_v_cost = std::make_shared<SemiquadraticCost>(
      kMaxVCostWeight, kP1VIdx, kMinV, !kOrientedRight, "MinV");
  const auto p1_max_v_cost = std::make_shared<SemiquadraticCost>(
      kMaxVCostWeight, kP1VIdx, kP1MaxV, kOrientedRight, "MaxV");
  const auto p1_nominal_v_cost = std::make_shared<QuadraticCost>(
      kNominalVCostWeight, kP1VIdx, kP1NominalV, "NominalV");
  p1_cost.AddStateCost(p1_min_v_cost);
  p1_cost.AddStateCost(p1_max_v_cost);
  p1_cost.AddStateCost(p1_nominal_v_cost);

  const auto p2_min_v_cost = std::make_shared<SemiquadraticCost>(
      kMaxVCostWeight, kP2VIdx, kMinV, !kOrientedRight, "MinV");
  const auto p2_max_v_cost = std::make_shared<SemiquadraticCost>(
      kMaxVCostWeight, kP2VIdx, kP2MaxV, kOrientedRight, "MaxV");
  const auto p2_nominal_v_cost = std::make_shared<QuadraticCost>(
      kNominalVCostWeight, kP2VIdx, kP2NominalV, "NominalV");
  p2_cost.AddStateCost(p2_min_v_cost);
  p2_cost.AddStateCost(p2_max_v_cost);
  p2_cost.AddStateCost(p2_nominal_v_cost);

  // Curvature costs for P0 and P1.
  const auto p0_curvature_cost = std::make_shared<QuadraticCost>(
      kCurvatureCostWeight, kP0KappaIdx, 0.0, "Curvature");
  p0_cost.AddStateCost(p0_curvature_cost);

  const auto p1_curvature_cost = std::make_shared<QuadraticCost>(
      kCurvatureCostWeight, kP1KappaIdx, 0.0, "Curvature");
  p1_cost.AddStateCost(p1_curvature_cost);

  // Penalize control effort.
  const auto p0_omega_cost = std::make_shared<QuadraticCost>(
      kOmegaCostWeight, kP0OmegaIdx, 0.0, "Steering");
  const auto p0_a_cost = std::make_shared<QuadraticCost>(kACostWeight, kP0AIdx,
                                                         0.0, "Acceleration");
  p0_cost.AddControlCost(0, p0_omega_cost);
  p0_cost.AddControlCost(0, p0_a_cost);

  const auto p1_omega_cost = std::make_shared<QuadraticCost>(
      kOmegaCostWeight, kP1OmegaIdx, 0.0, "Steering");
  const auto p1_a_cost = std::make_shared<QuadraticCost>(kACostWeight, kP1AIdx,
                                                         0.0, "Acceleration");
  p1_cost.AddControlCost(1, p1_omega_cost);
  p1_cost.AddControlCost(1, p1_a_cost);

  const auto p2_omega_cost = std::make_shared<QuadraticCost>(
      kOmegaCostWeight, kP2OmegaIdx, 0.0, "Steering");
  const auto p2_a_cost = std::make_shared<QuadraticCost>(kACostWeight, kP2AIdx,
                                                         0.0, "Acceleration");
  p2_cost.AddControlCost(2, p2_omega_cost);
  p2_cost.AddControlCost(2, p2_a_cost);

  // Path lenth costs.
  const auto p0_s_cost = std::make_shared<NominalPathLengthCost>(
      kSCostWeight, kP0SIdx, kP0NominalV, "PathLength");
  p0_cost.AddStateCost(p0_s_cost);

  const auto p1_s_cost = std::make_shared<NominalPathLengthCost>(
      kSCostWeight, kP1SIdx, kP1NominalV, "PathLenth");
  p1_cost.AddStateCost(p1_s_cost);

  const auto p2_s_cost = std::make_shared<NominalPathLengthCost>(
      kSCostWeight, kP2SIdx, kP2NominalV, "PathLength");
  p2_cost.AddStateCost(p2_s_cost);

  // Goal costs.
  const auto p0_goalx_cost = std::make_shared<FinalTimeCost>(
      std::make_shared<QuadraticCost>(kGoalCostWeight, kP0XIdx, kP0GoalX),
      kTimeHorizon - 1.0, "GoalX");
  const auto p0_goaly_cost = std::make_shared<FinalTimeCost>(
      std::make_shared<QuadraticCost>(kGoalCostWeight, kP0YIdx, kP0GoalY),
      kTimeHorizon - 1.0, "GoalY");
  p0_cost.AddStateCost(p0_goalx_cost);
  p0_cost.AddStateCost(p0_goaly_cost);

  const auto p1_goalx_cost = std::make_shared<FinalTimeCost>(
      std::make_shared<QuadraticCost>(kGoalCostWeight, kP1XIdx, kP1GoalX),
      kTimeHorizon - 1.0, "GoalX");
  const auto p1_goaly_cost = std::make_shared<FinalTimeCost>(
      std::make_shared<QuadraticCost>(kGoalCostWeight, kP1YIdx, kP1GoalY),
      kTimeHorizon - 1.0, "GoalY");
  p1_cost.AddStateCost(p1_goalx_cost);
  p1_cost.AddStateCost(p1_goaly_cost);

  const auto p2_goalx_cost = std::make_shared<FinalTimeCost>(
      std::make_shared<QuadraticCost>(kGoalCostWeight, kP2XIdx, kP2GoalX),
      kTimeHorizon - 1.0, "GoalX");
  const auto p2_goaly_cost = std::make_shared<FinalTimeCost>(
      std::make_shared<QuadraticCost>(kGoalCostWeight, kP2YIdx, kP2GoalY),
      kTimeHorizon - 1.0, "GoalY");
  p2_cost.AddStateCost(p2_goalx_cost);
  p2_cost.AddStateCost(p2_goaly_cost);

  // Pairwise proximity costs.
  const std::shared_ptr<ProximityCost> p0p1_proximity_cost(
      new ProximityCost(kP0ProximityCostWeight, {kP0XIdx, kP0YIdx},
                        {kP1XIdx, kP1YIdx}, "ProximityP1"));
  const std::shared_ptr<ProximityCost> p0p2_proximity_cost(
      new ProximityCost(kP0ProximityCostWeight, {kP0XIdx, kP0YIdx},
                        {kP2XIdx, kP2YIdx}, "ProximityP2"));
  p0_cost.AddStateCost(p0p1_proximity_cost);
  p0_cost.AddStateCost(p0p2_proximity_cost);

  const std::shared_ptr<ProximityCost> p1p0_proximity_cost(
      new ProximityCost(kP1ProximityCostWeight, {kP1XIdx, kP1YIdx},
                        {kP0XIdx, kP0YIdx}, "ProximityP0"));
  const std::shared_ptr<ProximityCost> p1p2_proximity_cost(
      new ProximityCost(kP1ProximityCostWeight, {kP1XIdx, kP1YIdx},
                        {kP2XIdx, kP2YIdx}, "ProximityP2"));
  p1_cost.AddStateCost(p1p0_proximity_cost);
  p1_cost.AddStateCost(p1p2_proximity_cost);

  const std::shared_ptr<ProximityCost> p2p0_proximity_cost(
      new ProximityCost(kP2ProximityCostWeight, {kP2XIdx, kP2YIdx},
                        {kP0XIdx, kP0YIdx}, "ProximityP0"));
  const std::shared_ptr<ProximityCost> p2p1_proximity_cost(
      new ProximityCost(kP2ProximityCostWeight, {kP2XIdx, kP2YIdx},
                        {kP1XIdx, kP1YIdx}, "ProximityP1"));
  p2_cost.AddStateCost(p2p0_proximity_cost);
  p2_cost.AddStateCost(p2p1_proximity_cost);

  // Set up solver.
  solver_.reset(new LinesearchingILQSolver(
      dynamics, {p0_cost, p1_cost, p2_cost}, kTimeHorizon, kTimeStep));
}

}  // namespace ilqgames
