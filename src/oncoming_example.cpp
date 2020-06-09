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
 *       from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE59;16M COPYRIGHT HOLDERS AND CONTRIBUTORS AS
 * IS AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
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
// Oncoming example.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/constraint/polyline2_signed_distance_constraint.h>
#include <ilqgames/constraint/proximity_constraint.h>
#include <ilqgames/constraint/single_dimension_constraint.h>
#include <ilqgames/cost/curvature_cost.h>
#include <ilqgames/cost/final_time_cost.h>
#include <ilqgames/cost/initial_time_cost.h>
#include <ilqgames/cost/locally_convex_proximity_cost.h>
#include <ilqgames/cost/nominal_path_length_cost.h>
#include <ilqgames/cost/orientation_cost.h>
#include <ilqgames/cost/proximity_cost.h>
#include <ilqgames/cost/quadratic_cost.h>
#include <ilqgames/cost/quadratic_difference_cost.h>
#include <ilqgames/cost/quadratic_polyline2_cost.h>
#include <ilqgames/cost/semiquadratic_cost.h>
#include <ilqgames/cost/semiquadratic_polyline2_cost.h>
#include <ilqgames/cost/weighted_convex_proximity_cost.h>
#include <ilqgames/dynamics/concatenated_dynamical_system.h>
#include <ilqgames/dynamics/single_player_car_6d.h>
#include <ilqgames/dynamics/single_player_unicycle_4d.h>
#include <ilqgames/examples/oncoming_example.h>
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

// // Adversarial time.
// DEFINE_double(adversarial_time, 0.0, "Adversarial time window (s).");

namespace ilqgames {

namespace {

// Time.
static constexpr Time kTimeStep = 0.1;     // s
static constexpr Time kTimeHorizon = 15.0; // s
static constexpr size_t kNumTimeSteps =
    static_cast<size_t>(kTimeHorizon / kTimeStep);

// Car inter-axle distance.
static constexpr float kInterAxleLength = 4.0; // m

// Cost weights.

static constexpr float kOmegaCostWeight = 50.0;
static constexpr float kJerkCostWeight = 50.0;

static constexpr float kACostWeight = 50.0;
static constexpr float kP1NominalVCostWeight = 10.0;
static constexpr float kP2NominalVCostWeight = 5.0;
// static constexpr float kP3NominalVCostWeight = 1.0;

// Newly added, 10-16-2019 20:33 p.m.
static constexpr float kMinV = 0.0;    // m/s
// static constexpr float kP1MaxV = 35.8; // m/s
// static constexpr float kP2MaxV = 35.8; // m/s

static constexpr float kP1MaxV = 10.0; // m/s
static constexpr float kP2MaxV = 10.0; // m/s

static constexpr float kLaneCostWeight = 10.0;
static constexpr float kLaneBoundaryCostWeight = 5.0;

static constexpr float kMinProximity = 3.0;
static constexpr float kP1ProximityCostWeight = 300.0;
static constexpr float kP2ProximityCostWeight = 1000.0;
// static constexpr float kP3ProximityCostWeight = 100.0;
using ProxCost = ProximityCost;

// Heading weight
static constexpr float kNominalHeadingCostWeight = 50.0;

static constexpr bool kOrientedRight = true;
static constexpr bool kConstraintOrientedInside = false;

// Lane width.
static constexpr float kLaneHalfWidth = 2.5; // m

// Nominal speed.
static constexpr float kP1NominalV = 5.0; // m/s
static constexpr float kP2NominalV = 5.0; // m/s

// Nominal heading
static constexpr float kP1NominalHeading = M_PI_2; // rad

// Initial state.
static constexpr float kP1InitialX = 1.5;   // m
static constexpr float kP1InitialY = -35.0; // m

static constexpr float kP2InitialX = -1.5;  // m
static constexpr float kP2InitialY = 35.0; // m
// static constexpr float kP2InitialYAntiparallel = 55.0; // m

static constexpr float kP1InitialHeading = M_PI_2;              // rad
static constexpr float kP2InitialHeading = -M_PI_2;             // rad
static constexpr float kP2InitialHeadingAntiparallel = -M_PI_2; // rad

static constexpr float kP1InitialSpeed = 5.1; // m/s
static constexpr float kP2InitialSpeed = 5.1; // m/s

// State dimensions.
using P1 = SinglePlayerCar6D;
using P2 = SinglePlayerCar6D;
using P3 = SinglePlayerCar6D;
// using P3 = SinglePlayerUnicycle4D;

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

// static const Dimension kP2XIdx = P1::kNumXDims + P2::kNumXDims +
// P2::kPxIdx; static const Dimension kP2YIdx = P1::kNumXDims + P2::kNumXDims
// + P2::kPyIdx; static const Dimension kP2HeadingIdx = P1::kNumXDims +
// P2::kNumXDims + P2::kThetaIdx; static const Dimension kP2PhiIdx =
// P1::kNumXDims + P2::kNumXDims + P2::kPhiIdx; static const Dimension kP2VIdx
// = P1::kNumXDims + P2::kNumXDims + P2::kVIdx; static const Dimension kP2AIdx
// = P1::kNumXDims + P2::kNumXDims + P2::kAIdx;

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
static const Dimension kP3JerkIdx = 1;
} // anonymous namespace

OncomingExample::OncomingExample(const SolverParams &params) {
  // Create dynamics.
  const std::shared_ptr<const ConcatenatedDynamicalSystem> dynamics(
      new ConcatenatedDynamicalSystem(
          {std::make_shared<SinglePlayerCar6D>(kInterAxleLength),
           std::make_shared<SinglePlayerCar6D>(kInterAxleLength)},
          kTimeStep));

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

  // if (scenario == "Parallel") {
  //   x0_(kP2HeadingIdx) = kP2InitialHeading;
  //   x0_(kP2YIdx) = kP2InitialY;
  // } else {
  //   x0_(kP2HeadingIdx) = kP2InitialHeadingAntiparallel;
  //   x0_(kP2YIdx) = kP2InitialYAntiparallel;
  // }

  x0_(kP2VIdx) = kP2InitialSpeed;

  // x0_(kP3XIdx) = kP3InitialX;
  // x0_(kP3YIdx) = kP3InitialY;
  // x0_(kP3HeadingIdx) = kP3InitialHeading;
  // x0_(kP3VIdx) = kP3InitialSpeed;

  // Set up initial strategies and operating point.
  strategies_.reset(new std::vector<Strategy>());
  for (PlayerIndex ii = 0; ii < dynamics->NumPlayers(); ii++)
    strategies_->emplace_back(kNumTimeSteps, dynamics->XDim(),
                              dynamics->UDim(ii));

  operating_point_.reset(
      new OperatingPoint(kNumTimeSteps, dynamics->NumPlayers(), 0.0, dynamics));

  // Set up costs for all players.
  PlayerCost p1_cost, p2_cost;

  // Orientation cost
  const auto p1_nominal_orientation_cost = std::make_shared<OrientationCost>(
      kNominalHeadingCostWeight, kP1HeadingIdx, kP1NominalHeading,
      "NominalHeadingP1");
  // p1_cost.AddStateCost(p1_nominal_orientation_cost);
  const auto p2_nominal_orientation_cost = std::make_shared<OrientationCost>(
      kNominalHeadingCostWeight, kP2HeadingIdx, kP1NominalHeading,
      "NominalHeadingP2");
  // p2_cost.AddStateCost(p2_nominal_orientation_cost);
  // const auto p3_nominal_orientation_cost =
  // std::make_shared<OrientationCost>(
  //     kNominalHeadingCostWeight, kP3HeadingIdx, kP1NominalHeading,
  // NominalHeadingP3");
  // p3_cost.AddStateCost(p3_nominal_orientation_cost);

  // Stay in lanes.

  const Polyline2 lane1(
      {Point2(kP1InitialX, -1000.0), Point2(kP1InitialX, 1000.0)});
  const Polyline2 lane2(
      {Point2(kP2InitialX, -1000.0), Point2(kP2InitialX, 1000.0)});

  const std::shared_ptr<QuadraticPolyline2Cost> p1_lane_cost(
      new QuadraticPolyline2Cost(kLaneCostWeight, lane1, {kP1XIdx, kP1YIdx},
                                 "LaneCenter"));
  const std::shared_ptr<Polyline2SignedDistanceConstraint> p1_lane_r_constraint(
      new Polyline2SignedDistanceConstraint(lane1, {kP1XIdx, kP1YIdx},
                                            -kLaneHalfWidth, kOrientedRight,
                                            "LaneRightBoundary"));
  const std::shared_ptr<Polyline2SignedDistanceConstraint> p1_lane_l_constraint(
      new Polyline2SignedDistanceConstraint(lane1, {kP1XIdx, kP1YIdx},
                                            kLaneHalfWidth, !kOrientedRight,
                                            "LaneLeftBoundary"));
  p1_cost.AddStateCost(p1_lane_cost);
  p1_cost.AddStateConstraint(p1_lane_r_constraint);
  p1_cost.AddStateConstraint(p1_lane_l_constraint);

  const std::shared_ptr<QuadraticPolyline2Cost> p2_lane_cost(
      new QuadraticPolyline2Cost(kLaneCostWeight, lane2, {kP2XIdx, kP2YIdx},
                                 "LaneCenter"));
  const std::shared_ptr<Polyline2SignedDistanceConstraint> p2_lane_r_constraint(
      new Polyline2SignedDistanceConstraint(lane2, {kP2XIdx, kP2YIdx},
                                            -kLaneHalfWidth, kOrientedRight,
                                            "LaneRightBoundary"));
  const std::shared_ptr<Polyline2SignedDistanceConstraint> p2_lane_l_constraint(
      new Polyline2SignedDistanceConstraint(lane2, {kP2XIdx, kP2YIdx},
                                            kLaneHalfWidth, !kOrientedRight,
                                            "LaneLeftBoundary"));
  p2_cost.AddStateCost(p2_lane_cost);
  p2_cost.AddStateConstraint(p2_lane_r_constraint);
  p2_cost.AddStateConstraint(p2_lane_l_constraint);

  // Max/min/nominal speed costs.

  const auto p1_min_v_constraint = std::make_shared<SingleDimensionConstraint>(
      kP1VIdx, kMinV, kOrientedRight, "MinV");
  const auto p1_max_v_constraint = std::make_shared<SingleDimensionConstraint>(
      kP1VIdx, kP1MaxV, !kOrientedRight, "MaxV");
  const auto p1_nominal_v_cost = std::make_shared<QuadraticCost>(
      kP1NominalVCostWeight, kP1VIdx, kP1NominalV, "NominalV");
  p1_cost.AddStateConstraint(p1_min_v_constraint);
  p1_cost.AddStateConstraint(p1_max_v_constraint);
  p1_cost.AddStateCost(p1_nominal_v_cost);

  const auto p2_min_v_constraint = std::make_shared<SingleDimensionConstraint>(
      kP2VIdx, kMinV, kOrientedRight, "MinV");
  const auto p2_max_v_constraint = std::make_shared<SingleDimensionConstraint>(
      kP2VIdx, kP2MaxV, !kOrientedRight, "MaxV");
  const auto p2_nominal_v_cost = std::make_shared<QuadraticCost>(
      kP2NominalVCostWeight, kP2VIdx, kP2NominalV, "NominalV");
  p2_cost.AddStateConstraint(p2_min_v_constraint);
  p2_cost.AddStateConstraint(p2_max_v_constraint);
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

  // Pairwise proximity costs.
  // const std::shared_ptr<ProxCost> p1p2_proximity_cost(
  //     new ProxCost(kP1ProximityCostWeight, {kP1XIdx, kP1YIdx},
  //                  {kP2XIdx, kP2YIdx}, kMinProximity, "ProximityP2"));
  // p1_cost.AddStateCost(p1p2_proximity_cost);

  // Collision-avoidance constraints.
  const std::shared_ptr<ProximityConstraint> p1p2_proximity_constraint(
      new ProximityConstraint({kP1XIdx, kP1YIdx}, {kP2XIdx, kP2YIdx},
                              kMinProximity, kConstraintOrientedInside,
                              "ProximityConstraintP2"));

  p1_cost.AddStateConstraint(p1p2_proximity_constraint);

  const std::shared_ptr<InitialTimeCost> p2p1_initial_proximity_cost(
      new InitialTimeCost(
          std::shared_ptr<QuadraticDifferenceCost>(new QuadraticDifferenceCost(
              kP2ProximityCostWeight, {kP2XIdx, kP2YIdx}, {kP1XIdx, kP1YIdx})),
          params.adversarial_time, "InitialProximityCostP1"));
  p2_cost.AddStateCost(p2p1_initial_proximity_cost);
  initial_time_costs_.push_back(p2p1_initial_proximity_cost);

  const std::shared_ptr<FinalTimeCost> p2p1_final_proximity_cost(
      new FinalTimeCost(std::shared_ptr<ProxCost>(new ProxCost(
                            kP2ProximityCostWeight, {kP2XIdx, kP2YIdx},
                            {kP1XIdx, kP1YIdx}, kMinProximity)),
                        params.adversarial_time, "FinalProximityCostP1"));
  p2_cost.AddStateCost(p2p1_final_proximity_cost);
  final_time_costs_.push_back(p2p1_final_proximity_cost);

  // Set up solver.
  solver_.reset(
      new ILQSolver(dynamics, {p1_cost, p2_cost}, kTimeHorizon, params));
}

inline std::vector<float> OncomingExample::Xs(const VectorXf &x) const {
  return {x(kP1XIdx), x(kP2XIdx)};
}

inline std::vector<float> OncomingExample::Ys(const VectorXf &x) const {
  return {x(kP1YIdx), x(kP2YIdx)};
}

inline std::vector<float> OncomingExample::Thetas(const VectorXf &x) const {
  return {x(kP1HeadingIdx), x(kP2HeadingIdx)};
}

} // namespace ilqgames
