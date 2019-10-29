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
 *          Chih-Yuan (Frank) Chiu (chihyuan_chiu@berkeley.edu)
 */

////////////////////////////////////////////////////////////////////////////////
//
// Originally: Three player intersection example. Ordering given by:
// (P1, P2, P3) = (Car 1, Car 2, Pedestrian).
//
// Now: Four player intersection example. Ordering given by:
// (P1, P2, P3, P4) = (Car 1, Car 2, Pedestrian 1, Pedestrian 2).
//
///////////////////////////////////////////////////////////////////////////////

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
#include <ilqgames/dynamics/single_player_car_6d.h>
#include <ilqgames/dynamics/single_player_unicycle_4d.h>
// #include <ilqgames/examples/three_player_intersection_example.h> // NEW
#include <ilqgames/examples/four_player_intersection_example.h> // NEW
#include <ilqgames/geometry/polyline2.h>
#include <ilqgames/solver/ilq_solver.h>
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
// Time.
static constexpr Time kTimeStep = 0.1;      // s
static constexpr Time kTimeHorizon = 10.0;  // s
static constexpr size_t kNumTimeSteps =
    static_cast<size_t>(kTimeHorizon / kTimeStep);

// Car inter-axle distance.
static constexpr float kInterAxleLength = 4.0;  // m

// Cost weights.
static constexpr float kOmegaCostWeight = 50.0;
static constexpr float kJerkCostWeight = 50.0;

static constexpr float kACostWeight = 5.0;
static constexpr float kCurvatureCostWeight = 1.0;
static constexpr float kMaxVCostWeight = 10.0;
static constexpr float kNominalVCostWeight = 10.0;

static constexpr float kGoalCostWeight = 0.1;
static constexpr float kLaneCostWeight = 25.0;
static constexpr float kLaneBoundaryCostWeight = 100.0;

static constexpr float kMinProximity = 6.0;
static constexpr float kP1ProximityCostWeight = 100.0;
static constexpr float kP2ProximityCostWeight = 100.0;
static constexpr float kP3ProximityCostWeight = 10.0;
static constexpr float kP4ProximityCostWeight = 10.0; // NEW
using ProxCost = ProximityCost;

static constexpr bool kOrientedRight = true;

// Lane width.
static constexpr float kLaneHalfWidth = 2.5;  // m

// Goal points.
static constexpr float kP1GoalX = -6.0;   // m
static constexpr float kP1GoalY = 600.0;  // m

static constexpr float kP2GoalX = 500.0;  // m
static constexpr float kP2GoalY = 12.0;   // m

static constexpr float kP3GoalX = 100.0;  // m
static constexpr float kP3GoalY = 16.0;   // m

static constexpr float kP4GoalX = 100.0;  // m NEW
static constexpr float kP4GoalY = 16.0;   // m NEW

// Nominal and max speed.
static constexpr float kP1MaxV = 12.0;  // m/s
static constexpr float kP2MaxV = 12.0;  // m/s
static constexpr float kP3MaxV = 2.0;   // m/s
static constexpr float kP4MaxV = 2.0;   // m/s NEW
static constexpr float kMinV = 1.0;     // m/s

static constexpr float kP1NominalV = 8.0;  // m/s
static constexpr float kP2NominalV = 5.0;  // m/s
static constexpr float kP3NominalV = 1.5;  // m/s
static constexpr float kP4NominalV = 1.5;  // m/s NEW

// Initial state.
static constexpr float kP1InitialX = -2.0;   // m
static constexpr float kP2InitialX = -10.0;  // m
static constexpr float kP3InitialX = -11.0;  // m
static constexpr float kP4InitialX = 0.0;  // m NEW

static constexpr float kP1InitialY = -30.0;  // m
static constexpr float kP2InitialY = 45.0;   // m
static constexpr float kP3InitialY = 16.0;   // m
static constexpr float kP4InitialY = 0.0;   // m

static constexpr float kP1InitialHeading = M_PI_2;  // rad
static constexpr float kP2InitialHeading = -M_PI_2; // rad
static constexpr float kP3InitialHeading = 0.0;     // rad
static constexpr float kP4InitialHeading = M_PI_2;  // rad NEW, changed heading

static constexpr float kP1InitialSpeed = 5.0;   // m/s
static constexpr float kP2InitialSpeed = 5.0;   // m/s
static constexpr float kP3InitialSpeed = 1.25;  // m/s
static constexpr float kP4InitialSpeed = 1;  // m/s NEW, slightly slower

// State dimensions.
using P1 = SinglePlayerCar6D;
using P2 = SinglePlayerCar6D;
using P3 = SinglePlayerUnicycle4D;
using P4 = SinglePlayerUnicycle4D; // NEW

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

// P4's X, Y, Heading, V Indices

static const Dimension kP4XIdx = // NEW
   P1::kNumXDims + P2::kNumXDims + P3::kNumXDims + P4::kPxIdx; // NEW
static const Dimension kP4YIdx = // NEW
   P1::kNumXDims + P2::kNumXDims + P3::kNumXDims + P4::kPyIdx; // NEW
static const Dimension kP4HeadingIdx = // NEW
   P1::kNumXDims + P2::kNumXDims + P3::kNumXDims + P4::kThetaIdx; // NEW
static const Dimension kP4VIdx = // NEW
   P1::kNumXDims + P2::kNumXDims + P3::kNumXDims + P4::kVIdx; // NEW


// Control dimensions.
static const Dimension kP1OmegaIdx = 0;
static const Dimension kP1JerkIdx = 1;
static const Dimension kP2OmegaIdx = 0;
static const Dimension kP2JerkIdx = 1;
static const Dimension kP3OmegaIdx = 0;
static const Dimension kP3AIdx = 1;
static const Dimension kP4OmegaIdx = 0; // NEW
static const Dimension kP4AIdx = 1; // NEW

}  // anonymous namespace

FourPlayerIntersectionExample::FourPlayerIntersectionExample(
    const SolverParams& params) {
  // Create dynamics.
  const std::shared_ptr<const ConcatenatedDynamicalSystem> dynamics(
      new ConcatenatedDynamicalSystem(
          {std::make_shared<SinglePlayerCar6D>(kInterAxleLength),
           std::make_shared<SinglePlayerCar6D>(kInterAxleLength),
           std::make_shared<SinglePlayerUnicycle4D>(),
           std::make_shared<SinglePlayerUnicycle4D>()},
          kTimeStep)); // NEW

  // Set up initial state.
  x0_ = VectorXf::Zero(dynamics->XDim());
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
  x0_(kP4XIdx) = kP4InitialX; // NEW
  x0_(kP4YIdx) = kP4InitialY; // NEW
  x0_(kP4HeadingIdx) = kP4InitialHeading; // NEW
  x0_(kP4VIdx) = kP4InitialSpeed; // NEW


  // Set up initial strategies and operating point.
  strategies_.reset(new std::vector<Strategy>());
  for (PlayerIndex ii = 0; ii < dynamics->NumPlayers(); ii++)
    strategies_->emplace_back(kNumTimeSteps, dynamics->XDim(),
                              dynamics->UDim(ii));

  operating_point_.reset(
      new OperatingPoint(kNumTimeSteps, dynamics->NumPlayers(), 0.0, dynamics));

  // Set up costs for all players.
  PlayerCost p1_cost, p2_cost, p3_cost, p4_cost; // NEW

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
  const Polyline2 lane4(
      {Point2(-1000.0, kP4InitialY), Point2(1000.0, kP4InitialY)}); // NEW LANE?

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

  const std::shared_ptr<QuadraticPolyline2Cost> p3_lane_cost(
      new QuadraticPolyline2Cost(kLaneCostWeight, lane3, {kP3XIdx, kP3YIdx},
                                 "LaneCenter"));
  const std::shared_ptr<SemiquadraticPolyline2Cost> p3_lane_r_cost(
      new SemiquadraticPolyline2Cost(kLaneBoundaryCostWeight, lane3,
                                     {kP3XIdx, kP3YIdx}, kLaneHalfWidth,
                                     kOrientedRight, "LaneRightBoundary"));
  const std::shared_ptr<SemiquadraticPolyline2Cost> p3_lane_l_cost(
      new SemiquadraticPolyline2Cost(kLaneBoundaryCostWeight, lane3,
                                     {kP3XIdx, kP3YIdx}, -kLaneHalfWidth,
                                     !kOrientedRight, "LaneLeftBoundary"));
  p3_cost.AddStateCost(p3_lane_cost);
  p3_cost.AddStateCost(p3_lane_r_cost);
  p3_cost.AddStateCost(p3_lane_l_cost);

  // NEW lane cost for P4

  const std::shared_ptr<QuadraticPolyline2Cost> p4_lane_cost(
      new QuadraticPolyline2Cost(kLaneCostWeight, lane4, {kP4XIdx, kP4YIdx},
                                 "LaneCenter"));
  const std::shared_ptr<SemiquadraticPolyline2Cost> p4_lane_r_cost(
      new SemiquadraticPolyline2Cost(kLaneBoundaryCostWeight, lane4,
                                     {kP4XIdx, kP4YIdx}, kLaneHalfWidth,
                                     kOrientedRight, "LaneRightBoundary"));
  const std::shared_ptr<SemiquadraticPolyline2Cost> p4_lane_l_cost(
      new SemiquadraticPolyline2Cost(kLaneBoundaryCostWeight, lane4,
                                     {kP4XIdx, kP4YIdx}, -kLaneHalfWidth,
                                     !kOrientedRight, "LaneLeftBoundary"));
  p4_cost.AddStateCost(p4_lane_cost);
  p4_cost.AddStateCost(p4_lane_r_cost);
  p4_cost.AddStateCost(p4_lane_l_cost);

  // Max/min/nominal speed costs.
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

  const auto p3_min_v_cost = std::make_shared<SemiquadraticCost>(
      kMaxVCostWeight, kP3VIdx, kMinV, !kOrientedRight, "MinV");
  const auto p3_max_v_cost = std::make_shared<SemiquadraticCost>(
      kMaxVCostWeight, kP3VIdx, kP3MaxV, kOrientedRight, "MaxV");
  const auto p3_nominal_v_cost = std::make_shared<QuadraticCost>(
      kNominalVCostWeight, kP3VIdx, kP3NominalV, "NominalV");
  p3_cost.AddStateCost(p3_min_v_cost);
  p3_cost.AddStateCost(p3_max_v_cost);
  p3_cost.AddStateCost(p3_nominal_v_cost);

  // NEW min_v_cost, max_v_cost, etc. for P4

  const auto p4_min_v_cost = std::make_shared<SemiquadraticCost>(
      kMaxVCostWeight, kP4VIdx, kMinV, !kOrientedRight, "MinV");
  const auto p4_max_v_cost = std::make_shared<SemiquadraticCost>(
      kMaxVCostWeight, kP4VIdx, kP4MaxV, kOrientedRight, "MaxV");
  const auto p4_nominal_v_cost = std::make_shared<QuadraticCost>(
      kNominalVCostWeight, kP4VIdx, kP4NominalV, "NominalV");
  p4_cost.AddStateCost(p3_min_v_cost);
  p4_cost.AddStateCost(p3_max_v_cost);
  p4_cost.AddStateCost(p3_nominal_v_cost);

  // // Curvature costs for P1 and P2.
  // const auto p1_curvature_cost = std::make_shared<QuadraticCost>(
  //     kCurvatureCostWeight, kP1PhiIdx, 0.0, "Curvature");
  // p1_cost.AddStateCost(p1_curvature_cost);

  // const auto p2_curvature_cost = std::make_shared<QuadraticCost>(
  //     kCurvatureCostWeight, kP2PhiIdx, 0.0, "Curvature");
  // p2_cost.AddStateCost(p2_curvature_cost);

  // // Penalize acceleration for cars.
  // const auto p1_a_cost = std::make_shared<QuadraticCost>(kACostWeight,
  // kP1AIdx,
  //                                                        0.0,
  //                                                        "Acceleration");
  // p1_cost.AddStateCost(p1_a_cost);

  // const auto p2_a_cost = std::make_shared<QuadraticCost>(kACostWeight,
  // kP2AIdx,
  //                                                        0.0,
  //                                                        "Acceleration");
  // p2_cost.AddStateCost(p2_a_cost);

  // Penalize control effort.
  const auto p1_omega_cost = std::make_shared<QuadraticCost>(
      kOmegaCostWeight, kP1OmegaIdx, 0.0, "Steering");
  const auto p1_jerk_cost =
      std::make_shared<QuadraticCost>(kJerkCostWeight, kP1JerkIdx, 0.0, "Jerk");
  p1_cost.AddControlCost(0, p1_omega_cost);
  p1_cost.AddControlCost(0, p1_jerk_cost);

  const auto p2_omega_cost = std::make_shared<QuadraticCost>(
      kOmegaCostWeight, kP2OmegaIdx, 0.0, "Steering");
  const auto p2_jerk_cost =
      std::make_shared<QuadraticCost>(kJerkCostWeight, kP2JerkIdx, 0.0, "Jerk");
  p2_cost.AddControlCost(1, p2_omega_cost);
  p2_cost.AddControlCost(1, p2_jerk_cost);

  const auto p3_omega_cost = std::make_shared<QuadraticCost>(
      kOmegaCostWeight, kP3OmegaIdx, 0.0, "Steering");
  const auto p3_a_cost = std::make_shared<QuadraticCost>(kACostWeight, kP3AIdx,
                                                         0.0, "Acceleration");
  p3_cost.AddControlCost(2, p3_omega_cost);
  p3_cost.AddControlCost(2, p3_a_cost);

  // NEW P4 omega_cost

  const auto p4_omega_cost = std::make_shared<QuadraticCost>(
      kOmegaCostWeight, kP4OmegaIdx, 0.0, "Steering");
  const auto p4_a_cost = std::make_shared<QuadraticCost>(kACostWeight, kP4AIdx,
                                                         0.0, "Acceleration");
  p4_cost.AddControlCost(3, p4_omega_cost);
  p4_cost.AddControlCost(3, p4_a_cost);

  // Goal costs.

  constexpr float kInitialTimeWindow = 0.5;  // s

  // to edit

  constexpr float kFinalTimeWindow = 0.5;  // s
  const auto p1_goalx_cost = std::make_shared<FinalTimeCost>(
      std::make_shared<QuadraticCost>(kGoalCostWeight, kP1XIdx, kP1GoalX),
      kTimeHorizon - kFinalTimeWindow, "GoalX");
  const auto p1_goaly_cost = std::make_shared<FinalTimeCost>(
      std::make_shared<QuadraticCost>(kGoalCostWeight, kP1YIdx, kP1GoalY),
      kTimeHorizon - kFinalTimeWindow, "GoalY");
  p1_cost.AddStateCost(p1_goalx_cost);
  p1_cost.AddStateCost(p1_goaly_cost);

  const auto p2_goalx_cost = std::make_shared<FinalTimeCost>(
      std::make_shared<QuadraticCost>(kGoalCostWeight, kP2XIdx, kP2GoalX),
      kTimeHorizon - kFinalTimeWindow, "GoalX");
  const auto p2_goaly_cost = std::make_shared<FinalTimeCost>(
      std::make_shared<QuadraticCost>(kGoalCostWeight, kP2YIdx, kP2GoalY),
      kTimeHorizon - kFinalTimeWindow, "GoalY");
  p2_cost.AddStateCost(p2_goalx_cost);
  p2_cost.AddStateCost(p2_goaly_cost);

  const auto p3_goalx_cost = std::make_shared<FinalTimeCost>(
      std::make_shared<QuadraticCost>(kGoalCostWeight, kP3XIdx, kP3GoalX),
      kTimeHorizon - kFinalTimeWindow, "GoalX");
  const auto p3_goaly_cost = std::make_shared<FinalTimeCost>(
      std::make_shared<QuadraticCost>(kGoalCostWeight, kP3YIdx, kP3GoalY),
      kTimeHorizon - kFinalTimeWindow, "GoalY");
  p3_cost.AddStateCost(p3_goalx_cost);
  p3_cost.AddStateCost(p3_goaly_cost);

  // NEW P4 goalx_cost, goaly_cost

  const auto p4_goalx_cost = std::make_shared<FinalTimeCost>(
      std::make_shared<QuadraticCost>(kGoalCostWeight, kP4XIdx, kP4GoalX),
      kTimeHorizon - kFinalTimeWindow, "GoalX");
  const auto p4_goaly_cost = std::make_shared<FinalTimeCost>(
      std::make_shared<QuadraticCost>(kGoalCostWeight, kP4YIdx, kP4GoalY),
      kTimeHorizon - kFinalTimeWindow, "GoalY");
  p4_cost.AddStateCost(p4_goalx_cost);
  p4_cost.AddStateCost(p4_goaly_cost);

  // Pairwise proximity costs.
  // New Pairwise proximity costs have been added for P1, P2, P3 w.r.t. P4.

  const std::shared_ptr<ProxCost> p1p2_proximity_cost(
      new ProxCost(kP1ProximityCostWeight, {kP1XIdx, kP1YIdx},
                   {kP2XIdx, kP2YIdx}, kMinProximity, "ProximityP2"));
  const std::shared_ptr<ProxCost> p1p3_proximity_cost(
      new ProxCost(kP1ProximityCostWeight, {kP1XIdx, kP1YIdx},
                   {kP3XIdx, kP3YIdx}, kMinProximity, "ProximityP3"));
  const std::shared_ptr<ProxCost> p1p4_proximity_cost(
      new ProxCost(kP1ProximityCostWeight, {kP1XIdx, kP1YIdx},
                   {kP4XIdx, kP4YIdx}, kMinProximity, "ProximityP4"));
  p1_cost.AddStateCost(p1p2_proximity_cost);
  p1_cost.AddStateCost(p1p3_proximity_cost);
  p1_cost.AddStateCost(p1p4_proximity_cost);

  const std::shared_ptr<ProxCost> p2p1_proximity_cost(
      new ProxCost(kP2ProximityCostWeight, {kP2XIdx, kP2YIdx},
                   {kP1XIdx, kP1YIdx}, kMinProximity, "ProximityP1"));
  const std::shared_ptr<ProxCost> p2p3_proximity_cost(
      new ProxCost(kP2ProximityCostWeight, {kP2XIdx, kP2YIdx},
                   {kP3XIdx, kP3YIdx}, kMinProximity, "ProximityP3"));
  const std::shared_ptr<ProxCost> p2p4_proximity_cost(
      new ProxCost(kP2ProximityCostWeight, {kP2XIdx, kP2YIdx},
                   {kP4XIdx, kP4YIdx}, kMinProximity, "ProximityP4"));
  p2_cost.AddStateCost(p2p1_proximity_cost);
  p2_cost.AddStateCost(p2p3_proximity_cost);
  p2_cost.AddStateCost(p2p4_proximity_cost);

  const std::shared_ptr<ProxCost> p3p1_proximity_cost(
      new ProxCost(kP3ProximityCostWeight, {kP3XIdx, kP3YIdx},
                   {kP1XIdx, kP1YIdx}, kMinProximity, "ProximityP1"));
  const std::shared_ptr<ProxCost> p3p2_proximity_cost(
      new ProxCost(kP3ProximityCostWeight, {kP3XIdx, kP3YIdx},
                   {kP2XIdx, kP2YIdx}, kMinProximity, "ProximityP2"));
  const std::shared_ptr<ProxCost> p3p4_proximity_cost(
      new ProxCost(kP3ProximityCostWeight, {kP3XIdx, kP3YIdx},
                   {kP4XIdx, kP4YIdx}, kMinProximity, "ProximityP4"));
  p3_cost.AddStateCost(p3p1_proximity_cost);
  p3_cost.AddStateCost(p3p2_proximity_cost);
  p3_cost.AddStateCost(p3p4_proximity_cost);

  // NEW P4 Pairwise Proximity Cost

  const std::shared_ptr<ProxCost> p4p1_proximity_cost(
      new ProxCost(kP4ProximityCostWeight, {kP4XIdx, kP4YIdx},
                   {kP1XIdx, kP1YIdx}, kMinProximity, "ProximityP1"));
  const std::shared_ptr<ProxCost> p4p2_proximity_cost(
      new ProxCost(kP4ProximityCostWeight, {kP4XIdx, kP4YIdx},
                   {kP2XIdx, kP2YIdx}, kMinProximity, "ProximityP2"));
  const std::shared_ptr<ProxCost> p4p3_proximity_cost(
      new ProxCost(kP4ProximityCostWeight, {kP4XIdx, kP4YIdx},
                   {kP3XIdx, kP3YIdx}, kMinProximity, "ProximityP3"));
  p4_cost.AddStateCost(p4p1_proximity_cost);
  p4_cost.AddStateCost(p4p2_proximity_cost);
  p4_cost.AddStateCost(p4p3_proximity_cost);

  // Set up solver.

  // Add in P4 costs, indices, etc.

  solver_.reset(new ILQSolver(dynamics, {p1_cost, p2_cost, p3_cost, p4_cost},
                              kTimeHorizon, params));
}

inline std::vector<float> FourPlayerIntersectionExample::Xs(
    const VectorXf& x) const {
  return {x(kP1XIdx), x(kP2XIdx), x(kP3XIdx), x(kP4XIdx)};
}

inline std::vector<float> FourPlayerIntersectionExample::Ys(
    const VectorXf& x) const {
  return {x(kP1YIdx), x(kP2YIdx), x(kP3YIdx), x(kP4YIdx)};
}

inline std::vector<float> FourPlayerIntersectionExample::Thetas(
    const VectorXf& x) const {
  return {x(kP1HeadingIdx), x(kP2HeadingIdx), x(kP3HeadingIdx),
          x(kP4HeadingIdx)};
}

}  // namespace ilqgames
