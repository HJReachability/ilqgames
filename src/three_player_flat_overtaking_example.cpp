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
// Three player *flat* overtaking example. Ordering is given by the following:
// (P1, P2, P3) = (Car 1, Car 2, Pedestrian).
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/cost/curvature_cost.h>
#include <ilqgames/cost/final_time_cost.h>
#include <ilqgames/cost/locally_convex_proximity_cost.h>
#include <ilqgames/cost/nominal_path_length_cost.h>
#include <ilqgames/cost/proximity_cost.h>
#include <ilqgames/cost/quadratic_cost.h>
#include <ilqgames/cost/quadratic_norm_cost.h>
#include <ilqgames/cost/quadratic_polyline2_cost.h>
#include <ilqgames/cost/route_progress_cost.h>
#include <ilqgames/cost/semiquadratic_cost.h>
#include <ilqgames/cost/semiquadratic_norm_cost.h>
#include <ilqgames/cost/semiquadratic_polyline2_cost.h>
#include <ilqgames/dynamics/concatenated_flat_system.h>
#include <ilqgames/dynamics/single_player_flat_car_6d.h>
#include <ilqgames/dynamics/single_player_flat_unicycle_4d.h>
#include <ilqgames/examples/three_player_flat_overtaking_example.h>
#include <ilqgames/geometry/polyline2.h>
#include <ilqgames/solver/problem.h>
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
static constexpr float kCarAuxCostWeight = 5000.0;

static constexpr float kP1NominalVCostWeight = 10.0;
static constexpr float kP2NominalVCostWeight = 1.0;
static constexpr float kP3NominalVCostWeight = 1.0;

static constexpr float kLaneCostWeight = 25.0;
static constexpr float kLaneBoundaryCostWeight = 100.0;

static constexpr float kMinProximity = 5.0;
static constexpr float kP1ProximityCostWeight = 100.0;
static constexpr float kP2ProximityCostWeight = 100.0;
static constexpr float kP3ProximityCostWeight = 100.0;
using ProxCost = ProximityCost;

// Heading weight
static constexpr float kNominalHeadingCostWeight = 150.0;

static constexpr bool kOrientedRight = true;

// Lane width.
static constexpr float kLaneHalfWidth = 2.5;  // m

// Nominal speeds.
static constexpr float kP1NominalV = 15.0;  // m/s
static constexpr float kP2NominalV = 10.0;  // m/s
static constexpr float kP3NominalV = 10.0;  // m/s

// Nominal heading
static constexpr float kP1NominalHeading = M_PI_2;  // rad

// Initial state.
static constexpr float kP1InitialX = 2.5;    // m
static constexpr float kP1InitialY = -10.0;  // m

static constexpr float kP2InitialX = -1.0;   // m
static constexpr float kP2InitialY = -10.0;  // m

static constexpr float kP3InitialX = 2.5;   // m
static constexpr float kP3InitialY = 10.0;  // m

static constexpr float kP1InitialHeading = M_PI_2;  // rad
static constexpr float kP2InitialHeading = M_PI_2;  // rad
static constexpr float kP3InitialHeading = M_PI_2;  // rad

static constexpr float kP1InitialSpeed = 5.0;   // m/s
static constexpr float kP2InitialSpeed = 5.0;   // m/s
static constexpr float kP3InitialSpeed = 5.25;  // m/s

// State dimensions.
using P1 = SinglePlayerFlatCar6D;
using P2 = SinglePlayerFlatCar6D;
using P3 = SinglePlayerFlatCar6D;

static const Dimension kP1XIdx = P1::kPxIdx;
static const Dimension kP1YIdx = P1::kPyIdx;
static const Dimension kP1HeadingIdx = P1::kThetaIdx;
static const Dimension kP1PhiIdx = P1::kPhiIdx;
static const Dimension kP1VIdx = P1::kVIdx;
static const Dimension kP1AIdx = P1::kAIdx;
static const Dimension kP1VxIdx = P1::kVxIdx;
static const Dimension kP1VyIdx = P1::kVyIdx;
static const Dimension kP1AxIdx = P1::kAxIdx;
static const Dimension kP1AyIdx = P1::kAyIdx;

static const Dimension kP2XIdx = P1::kNumXDims + P2::kPxIdx;
static const Dimension kP2YIdx = P1::kNumXDims + P2::kPyIdx;
static const Dimension kP2HeadingIdx = P1::kNumXDims + P2::kThetaIdx;
static const Dimension kP2PhiIdx = P1::kNumXDims + P2::kPhiIdx;
static const Dimension kP2VIdx = P1::kNumXDims + P2::kVIdx;
static const Dimension kP2AIdx = P1::kNumXDims + P2::kAIdx;
static const Dimension kP2VxIdx = P1::kNumXDims + P2::kVxIdx;
static const Dimension kP2VyIdx = P1::kNumXDims + P2::kVyIdx;
static const Dimension kP2AxIdx = P1::kNumXDims + P2::kAxIdx;
static const Dimension kP2AyIdx = P1::kNumXDims + P2::kAyIdx;

static const Dimension kP3XIdx = P1::kNumXDims + P2::kNumXDims + P3::kPxIdx;
static const Dimension kP3YIdx = P1::kNumXDims + P2::kNumXDims + P3::kPyIdx;
static const Dimension kP3HeadingIdx =
    P1::kNumXDims + P2::kNumXDims + P3::kThetaIdx;
static const Dimension kP3VIdx = P1::kNumXDims + P2::kNumXDims + P3::kVIdx;
static const Dimension kP3VxIdx = P1::kNumXDims + P2::kNumXDims + P3::kVxIdx;
static const Dimension kP3VyIdx = P1::kNumXDims + P2::kNumXDims + P3::kVyIdx;

// Control dimensions.
static const Dimension kP1OmegaIdx = 0;
static const Dimension kP1JerkIdx = 1;
static const Dimension kP2OmegaIdx = 0;
static const Dimension kP2JerkIdx = 1;
static const Dimension kP3OmegaIdx = 0;
static const Dimension kP3AIdx = 1;

}  // anonymous namespace

void ThreePlayerFlatOvertakingExample::ConstructDynamics() {
  dynamics_.reset(
      new ConcatenatedFlatSystem({std::make_shared<P1>(kInterAxleLength),
                                  std::make_shared<P2>(kInterAxleLength),
                                  std::make_shared<P3>(kInterAxleLength)}));
}

void ThreePlayerFlatOvertakingExample::ConstructInitialState() {
  VectorXf x0 = VectorXf::Zero(dynamics_->XDim());
  x0(kP1XIdx) = kP1InitialX;
  x0(kP1YIdx) = kP1InitialY;
  x0(kP1HeadingIdx) = kP1InitialHeading;
  x0(kP1VIdx) = kP1InitialSpeed;
  x0(kP2XIdx) = kP2InitialX;
  x0(kP2YIdx) = kP2InitialY;
  x0(kP2HeadingIdx) = kP2InitialHeading;
  x0(kP2VIdx) = kP2InitialSpeed;
  x0(kP3XIdx) = kP3InitialX;
  x0(kP3YIdx) = kP3InitialY;
  x0(kP3HeadingIdx) = kP3InitialHeading;
  x0(kP3VIdx) = kP3InitialSpeed;

  x0_ = dynamics_->ToLinearSystemState(x0);
}

void ThreePlayerFlatOvertakingExample::ConstructPlayerCosts() {
  // Set up costs for all players.
  player_costs_.emplace_back("P1");
  player_costs_.emplace_back("P2");
  player_costs_.emplace_back("P3");
  auto& p1_cost = player_costs_[0];
  auto& p2_cost = player_costs_[1];
  auto& p3_cost = player_costs_[2];

  // Stay in lanes.
  const Polyline2 lane1(
      {Point2(kP2InitialX, kP2InitialY), Point2(kP2InitialX, 1000.0)});
  const Polyline2 lane2(
      {Point2(kP3InitialX, kP3InitialY), Point2(kP3InitialX, 1000.0)});

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
      new QuadraticPolyline2Cost(kLaneCostWeight, lane1, {kP2XIdx, kP2YIdx},
                                 "LaneCenter"));
  const std::shared_ptr<SemiquadraticPolyline2Cost> p2_lane_r_cost(
      new SemiquadraticPolyline2Cost(kLaneBoundaryCostWeight, lane1,
                                     {kP2XIdx, kP2YIdx}, kLaneHalfWidth,
                                     kOrientedRight, "LaneRightBoundary"));
  const std::shared_ptr<SemiquadraticPolyline2Cost> p2_lane_l_cost(
      new SemiquadraticPolyline2Cost(kLaneBoundaryCostWeight, lane1,
                                     {kP2XIdx, kP2YIdx}, -kLaneHalfWidth,
                                     !kOrientedRight, "LaneLeftBoundary"));
  p2_cost.AddStateCost(p2_lane_cost);
  p2_cost.AddStateCost(p2_lane_r_cost);
  p2_cost.AddStateCost(p2_lane_l_cost);

  const std::shared_ptr<QuadraticPolyline2Cost> p3_lane_cost(
      new QuadraticPolyline2Cost(kLaneCostWeight, lane2, {kP3XIdx, kP3YIdx},
                                 "LaneCenter"));
  const std::shared_ptr<SemiquadraticPolyline2Cost> p3_lane_r_cost(
      new SemiquadraticPolyline2Cost(kLaneBoundaryCostWeight, lane2,
                                     {kP3XIdx, kP3YIdx}, kLaneHalfWidth,
                                     kOrientedRight, "LaneRightBoundary"));
  const std::shared_ptr<SemiquadraticPolyline2Cost> p3_lane_l_cost(
      new SemiquadraticPolyline2Cost(kLaneBoundaryCostWeight, lane2,
                                     {kP3XIdx, kP3YIdx}, -kLaneHalfWidth,
                                     !kOrientedRight, "LaneLeftBoundary"));
  p3_cost.AddStateCost(p3_lane_cost);
  p3_cost.AddStateCost(p3_lane_r_cost);
  p3_cost.AddStateCost(p3_lane_l_cost);

  // Max/min/nominal speed costs.
  const std::shared_ptr<RouteProgressCost> p1_progress_cost(
      new RouteProgressCost(kP1NominalVCostWeight, kP1NominalV, lane1,
                            {kP1XIdx, kP1YIdx}, "RouteProgress",
                            kP1InitialY - kP2InitialY));
  p1_cost.AddStateCost(p1_progress_cost);

  const std::shared_ptr<RouteProgressCost> p2_progress_cost(
      new RouteProgressCost(kP2NominalVCostWeight, kP2NominalV, lane1,
                            {kP2XIdx, kP2YIdx}, "RouteProgress"));
  p2_cost.AddStateCost(p2_progress_cost);

  const std::shared_ptr<RouteProgressCost> p3_progress_cost(
      new RouteProgressCost(kP3NominalVCostWeight, kP3NominalV, lane2,
                            {kP3XIdx, kP3YIdx}, "RouteProgress"));
  p3_cost.AddStateCost(p3_progress_cost);

  // Penalize control effort.
  constexpr Dimension kApplyInAllDimensions = -1;
  // const auto unicycle_aux_cost = std::make_shared<QuadraticCost>(
  //     kUnicycleAuxCostWeight, kApplyInAllDimensions, 0.0, "Auxiliary Input");
  const auto car_aux_cost = std::make_shared<QuadraticCost>(
      kCarAuxCostWeight, kApplyInAllDimensions, 0.0, "Auxiliary Input");
  p1_cost.AddControlCost(0, car_aux_cost);
  p2_cost.AddControlCost(1, car_aux_cost);
  p3_cost.AddControlCost(2, car_aux_cost);

  // Pairwise proximity costs.
  const std::shared_ptr<ProxCost> p1p2_proximity_cost(
      new ProxCost(kP1ProximityCostWeight, {kP1XIdx, kP1YIdx},
                   {kP2XIdx, kP2YIdx}, kMinProximity, "ProximityP2"));
  const std::shared_ptr<ProxCost> p1p3_proximity_cost(
      new ProxCost(kP1ProximityCostWeight, {kP1XIdx, kP1YIdx},
                   {kP3XIdx, kP3YIdx}, kMinProximity, "ProximityP3"));
  p1_cost.AddStateCost(p1p2_proximity_cost);
  p1_cost.AddStateCost(p1p3_proximity_cost);

  const std::shared_ptr<ProxCost> p2p1_proximity_cost(
      new ProxCost(kP2ProximityCostWeight, {kP2XIdx, kP2YIdx},
                   {kP1XIdx, kP1YIdx}, kMinProximity, "ProximityP1"));
  const std::shared_ptr<ProxCost> p2p3_proximity_cost(
      new ProxCost(kP2ProximityCostWeight, {kP2XIdx, kP2YIdx},
                   {kP3XIdx, kP3YIdx}, kMinProximity, "ProximityP3"));
  p2_cost.AddStateCost(p2p1_proximity_cost);
  p2_cost.AddStateCost(p2p3_proximity_cost);

  const std::shared_ptr<ProxCost> p3p1_proximity_cost(
      new ProxCost(kP3ProximityCostWeight, {kP3XIdx, kP3YIdx},
                   {kP1XIdx, kP1YIdx}, kMinProximity, "ProximityP1"));
  const std::shared_ptr<ProxCost> p3p2_proximity_cost(
      new ProxCost(kP3ProximityCostWeight, {kP3XIdx, kP3YIdx},
                   {kP2XIdx, kP2YIdx}, kMinProximity, "ProximityP2"));
  p3_cost.AddStateCost(p3p1_proximity_cost);
  p3_cost.AddStateCost(p3p2_proximity_cost);
}

inline std::vector<float> ThreePlayerFlatOvertakingExample::Xs(
    const VectorXf& xi) const {
  return {xi(kP1XIdx), xi(kP2XIdx), xi(kP3XIdx)};
}

inline std::vector<float> ThreePlayerFlatOvertakingExample::Ys(
    const VectorXf& xi) const {
  return {xi(kP1YIdx), xi(kP2YIdx), xi(kP3YIdx)};
}

inline std::vector<float> ThreePlayerFlatOvertakingExample::Thetas(
    const VectorXf& xi) const {
  const VectorXf x = dynamics_->FromLinearSystemState(xi);
  return {x(kP1HeadingIdx), x(kP2HeadingIdx), x(kP3HeadingIdx)};
}

}  // namespace ilqgames
