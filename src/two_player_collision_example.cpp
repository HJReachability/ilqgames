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
 * Authors: Tanmay Gautam          ( tgautam23@berkeley.edu )
 *          David Fridovich-Keil   ( dfk@eecs.berkeley.edu )
 */

///////////////////////////////////////////////////////////////////////////////
//
// Three player collision example.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/cost/curvature_cost.h>
#include <ilqgames/cost/final_time_cost.h>
#include <ilqgames/cost/locally_convex_proximity_cost.h>
#include <ilqgames/cost/nominal_path_length_cost.h>
#include <ilqgames/cost/orientation_cost.h>
#include <ilqgames/cost/proximity_cost.h>
#include <ilqgames/cost/quadratic_cost.h>
#include <ilqgames/cost/quadratic_polyline2_cost.h>
#include <ilqgames/cost/semiquadratic_cost.h>
#include <ilqgames/cost/semiquadratic_polyline2_cost.h>
#include <ilqgames/cost/weighted_convex_proximity_cost.h>
#include <ilqgames/dynamics/concatenated_dynamical_system.h>
#include <ilqgames/dynamics/single_player_car_6d.h>
#include <ilqgames/dynamics/single_player_unicycle_4d.h>
#include <ilqgames/examples/two_player_collision_example.h>
#include <ilqgames/geometry/polyline2.h>
#include <ilqgames/solver/ilq_solver.h>
#include <ilqgames/solver/lq_feedback_solver.h>
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
static constexpr float kOmegaCostWeight = 5000.0;
static constexpr float kJerkCostWeight = 3250.0;

static constexpr float kACostWeight = 50.0;
static constexpr float kP1NominalVCostWeight = 10.0;
static constexpr float kP2NominalVCostWeight = 1.0;

static constexpr float kLaneCostWeight = 250.0;
static constexpr float kLaneBoundaryCostWeight = 50000.0;

static constexpr float kMinProximity = 7.5;
static constexpr float kP1ProximityCostWeight = 5000.0;
static constexpr float kP2ProximityCostWeight = 5000.0;
using ProxCost = ProximityCost;

// Heading weight
static constexpr float kNominalHeadingCostWeight = 150.0;

static constexpr bool kOrientedRight = true;

// Lane width.
static constexpr float kLaneHalfWidth = 2.5;  // m

// Nominal speed.
static constexpr float kP1NominalV = 5.0;  // m/s
static constexpr float kP2NominalV = 5.0;  // m/s

// Nominal heading
static constexpr float kP1NominalHeading = M_PI_2;  // rad

// Initial state.
static constexpr float kP1InitialX = 2.5;    // m
static constexpr float kP1InitialY = -50.0;  // m

static constexpr float kP2InitialX = 2.5;   // m
static constexpr float kP2InitialY = 50.0;  // m

static constexpr float kP1InitialHeading = M_PI_2;   // rad
static constexpr float kP2InitialHeading = -M_PI_2;  // rad

static constexpr float kP1InitialSpeed = 10.0;  // m/s
static constexpr float kP2InitialSpeed = 2.0;   // m/s

// Goal cost weights
static constexpr float kGoalCostWeight = 1000.0;
static constexpr float kP1GoalX = 2.5;
static constexpr float kP1GoalY = 50.0;
static constexpr float kP2GoalX = 2.5;
static constexpr float kP2GoalY = -50.0;

// State dimensions.
using P1 = SinglePlayerCar6D;
using P2 = SinglePlayerCar6D;

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

// Control dimensions.
static const Dimension kP1OmegaIdx = 0;
static const Dimension kP1JerkIdx = 1;
static const Dimension kP2OmegaIdx = 0;
static const Dimension kP2JerkIdx = 1;

}  // anonymous namespace

void TwoPlayerCollisionExample::ConstructDynamics() {
  dynamics_.reset(new ConcatenatedDynamicalSystem(
      {std::make_shared<SinglePlayerCar6D>(kInterAxleLength),
       std::make_shared<SinglePlayerCar6D>(kInterAxleLength)}));
}

void TwoPlayerCollisionExample::ConstructInitialState() {
  x0_ = VectorXf::Zero(dynamics_->XDim());
  x0_(kP1XIdx) = kP1InitialX;
  x0_(kP1YIdx) = kP1InitialY;
  x0_(kP1HeadingIdx) = kP1InitialHeading;
  x0_(kP1VIdx) = kP1InitialSpeed;
  x0_(kP2XIdx) = kP2InitialX;
  x0_(kP2YIdx) = kP2InitialY;
  x0_(kP2HeadingIdx) = kP2InitialHeading;
  x0_(kP2VIdx) = kP2InitialSpeed;
}

void TwoPlayerCollisionExample::ConstructPlayerCosts() {
  // Set up costs for all players.
  player_costs_.emplace_back("P1", 1.0, 0.0);
  player_costs_.emplace_back("P2", 1.0, 0.0);
  auto& p1_cost = player_costs_[0];
  auto& p2_cost = player_costs_[1];

  // cost for deviating from the center of the lane (for both p1 and p2)
  const Polyline2 lane1_p1p2({Point2(2.5, -50.0), Point2(2.5, 50.0)});
  const std::shared_ptr<QuadraticPolyline2Cost> p1_lane1_cost(
      new QuadraticPolyline2Cost(kLaneCostWeight, lane1_p1p2,
                                 {kP1XIdx, kP1YIdx}, "LaneCenter"));
  const std::shared_ptr<QuadraticPolyline2Cost> p2_lane1_cost(
      new QuadraticPolyline2Cost(kLaneCostWeight * 10, lane1_p1p2,
                                 {kP2XIdx, kP2YIdx}, "LaneCenter"));
  p1_cost.AddStateCost(p1_lane1_cost);
  p2_cost.AddStateCost(p2_lane1_cost);

  // cost for leaving the left boundary of the lane (for both p1 and p2)
  const std::shared_ptr<SemiquadraticPolyline2Cost> p1_lane1_l_cost(
      new SemiquadraticPolyline2Cost(kLaneBoundaryCostWeight * 1000, lane1_p1p2,
                                     {kP1XIdx, kP1YIdx}, -kLaneHalfWidth,
                                     !kOrientedRight, "LaneLeftBoundary"));
  const std::shared_ptr<SemiquadraticPolyline2Cost> p2_lane1_l_cost(
      new SemiquadraticPolyline2Cost(kLaneBoundaryCostWeight * 10, lane1_p1p2,
                                     {kP2XIdx, kP2YIdx}, -kLaneHalfWidth,
                                     !kOrientedRight, "LaneLeftBoundary"));
  p1_cost.AddStateCost(p1_lane1_l_cost);
  p2_cost.AddStateCost(p2_lane1_l_cost);

  // p2 cost of leaving right lane boundary
  const std::shared_ptr<SemiquadraticPolyline2Cost> p2_lane1_r_cost(
      new SemiquadraticPolyline2Cost(kLaneBoundaryCostWeight, lane1_p1p2,
                                     {kP2XIdx, kP2YIdx}, kLaneHalfWidth,
                                     kOrientedRight, "LaneRightBoundary"));
  p2_cost.AddStateCost(p2_lane1_r_cost);

  // p1 right lane boundary cost
  const Polyline2 lane1_p1({Point2(2.5 + kLaneHalfWidth, -50.0),
                            Point2(2.5 + kLaneHalfWidth, -5.0)});
  const Polyline2 lane2_p1(
      {Point2(2.5 + kLaneHalfWidth, 5.0), Point2(2.5 + kLaneHalfWidth, 50.0)});
  const Polyline2 lane3_p1({Point2(10.0, -5.0), Point2(10.0, 5.0)});
  const Polyline2 lane4_p1(
      {Point2(2.5 + kLaneHalfWidth, 5.0), Point2(25.0, 5.0)});
  const Polyline2 lane5_p1(
      {Point2(2.5 + kLaneHalfWidth, -5.0), Point2(25, -5.0)});
  const std::shared_ptr<SemiquadraticPolyline2Cost> p1_lane2_r_cost(
      new SemiquadraticPolyline2Cost(kLaneBoundaryCostWeight, lane1_p1,
                                     {kP1XIdx, kP1YIdx}, 0.0, kOrientedRight,
                                     "LaneRightBoundary_lane1_p1"));
  const std::shared_ptr<SemiquadraticPolyline2Cost> p1_lane3_r_cost(
      new SemiquadraticPolyline2Cost(kLaneBoundaryCostWeight, lane2_p1,
                                     {kP1XIdx, kP1YIdx}, 0.0, kOrientedRight,
                                     "LaneRightBoundary_lane2_p1"));
  const std::shared_ptr<SemiquadraticPolyline2Cost> p1_lane4_r_cost(
      new SemiquadraticPolyline2Cost(kLaneBoundaryCostWeight, lane3_p1,
                                     {kP1XIdx, kP1YIdx}, 0.0, kOrientedRight,
                                     "LaneRightBoundary_lane3_p1"));
  const std::shared_ptr<SemiquadraticPolyline2Cost> p1_lane5_l_cost(
      new SemiquadraticPolyline2Cost(kLaneBoundaryCostWeight, lane4_p1,
                                     {kP1XIdx, kP1YIdx}, 0.0, !kOrientedRight,
                                     "LaneLeftBoundary_lane4_p1"));
  const std::shared_ptr<SemiquadraticPolyline2Cost> p1_lane6_r_cost(
      new SemiquadraticPolyline2Cost(kLaneBoundaryCostWeight, lane5_p1,
                                     {kP1XIdx, kP1YIdx}, 0.0, kOrientedRight,
                                     "LaneRightBoundary_lane5_p1"));
  p1_cost.AddStateCost(p1_lane2_r_cost);
  p1_cost.AddStateCost(p1_lane3_r_cost);
  p1_cost.AddStateCost(p1_lane4_r_cost);
  p1_cost.AddStateCost(p1_lane5_l_cost);
  p1_cost.AddStateCost(p1_lane6_r_cost);

  // Max/min/nominal speed costs.
  // const auto p1_min_v_cost = std::make_shared<SemiquadraticCost>(
  //     kMaxVCostWeight, kP1VIdx, kMinV, !kOrientedRight, "MinV");
  // const auto p1_max_v_cost = std::make_shared<SemiquadraticCost>(
  //     kMaxVCostWeight, kP1VIdx, kP1MaxV, kOrientedRight, "MaxV");
  const auto p1_nominal_v_cost = std::make_shared<QuadraticCost>(
      kP1NominalVCostWeight, kP1VIdx, kP1NominalV, "NominalV");
  // p1_cost.AddStateCost(p1_min_v_cost);
  // p1_cost.AddStateCost(p1_max_v_cost);
  p1_cost.AddStateCost(p1_nominal_v_cost);

  // const auto p2_min_v_cost = std::make_shared<SemiquadraticCost>(
  //     kMaxVCostWeight, kP2VIdx, kMinV, !kOrientedRight, "MinV");
  // const auto p2_max_v_cost = std::make_shared<SemiquadraticCost>(
  //     kMaxVCostWeight, kP2VIdx, kP2MaxV, kOrientedRight, "MaxV");
  const auto p2_nominal_v_cost = std::make_shared<QuadraticCost>(
      kP2NominalVCostWeight, kP2VIdx, kP2NominalV, "NominalV");
  // p2_cost.AddStateCost(p2_min_v_cost);
  // p2_cost.AddStateCost(p2_max_v_cost);
  p2_cost.AddStateCost(p2_nominal_v_cost);

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

  // Goal costs.
  constexpr float kFinalTimeWindow = 0.5;  // s
  const auto p1_goalx_cost = std::make_shared<FinalTimeCost>(
      std::make_shared<QuadraticCost>(kGoalCostWeight, kP1XIdx, kP1GoalX),
      time::kTimeHorizon - kFinalTimeWindow, "GoalX");
  const auto p1_goaly_cost = std::make_shared<FinalTimeCost>(
      std::make_shared<QuadraticCost>(kGoalCostWeight, kP1YIdx, kP1GoalY),
      time::kTimeHorizon - kFinalTimeWindow, "GoalY");
  p1_cost.AddStateCost(p1_goalx_cost);
  p1_cost.AddStateCost(p1_goaly_cost);

  const auto p2_goalx_cost = std::make_shared<FinalTimeCost>(
      std::make_shared<QuadraticCost>(kGoalCostWeight, kP2XIdx, kP2GoalX),
      time::kTimeHorizon - kFinalTimeWindow, "GoalX");
  const auto p2_goaly_cost = std::make_shared<FinalTimeCost>(
      std::make_shared<QuadraticCost>(kGoalCostWeight, kP2YIdx, kP2GoalY),
      time::kTimeHorizon - kFinalTimeWindow, "GoalY");
  p2_cost.AddStateCost(p2_goalx_cost);
  p2_cost.AddStateCost(p2_goaly_cost);

  // Pairwise proximity costs.
  const std::shared_ptr<ProxCost> p1p2_proximity_cost(
      new ProxCost(kP1ProximityCostWeight, {kP1XIdx, kP1YIdx},
                   {kP2XIdx, kP2YIdx}, kMinProximity, "ProximityP2"));
  p1_cost.AddStateCost(p1p2_proximity_cost);

  const std::shared_ptr<ProxCost> p2p1_proximity_cost(
      new ProxCost(kP2ProximityCostWeight, {kP2XIdx, kP2YIdx},
                   {kP1XIdx, kP1YIdx}, kMinProximity, "ProximityP1"));
  p2_cost.AddStateCost(p2p1_proximity_cost);
}

inline std::vector<float> TwoPlayerCollisionExample::Xs(
    const VectorXf& x) const {
  return {x(kP1XIdx), x(kP2XIdx)};
}

inline std::vector<float> TwoPlayerCollisionExample::Ys(
    const VectorXf& x) const {
  return {x(kP1YIdx), x(kP2YIdx)};
}

inline std::vector<float> TwoPlayerCollisionExample::Thetas(
    const VectorXf& x) const {
  return {x(kP1HeadingIdx), x(kP2HeadingIdx)};
}

}  // namespace ilqgames
