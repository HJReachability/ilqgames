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
// Simple skeleton example intended as a quick-start guide for learning to use
// this repository. This file is extensively commented; however, if you do have
// any questions please don't hesitate to post an issue on the repository
// (https://github.com/HJReachability/ilqgames/issues) or contact the author.
//
// Note: Throughout this file, if any of the function or class interfaces are
// unclear (e.g., which parameters are supposed to be used where), please do
// consult the relevant header or auto-generated documentation available at
// https://hjreachability.github.io/ilqgames/documentation/html/.
//
// Steps (some of these are already done in this example but can be modified as
// suggested below):
// 1. Set the cost weights.
// 2. Set the nominal speed for each car.
// 3. Add input constraints.
// 4. (Advanced) Try other constraints.
// 5. (Advanced) Add in a third player.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/constraint/polyline2_signed_distance_constraint.h>
#include <ilqgames/constraint/single_dimension_constraint.h>
#include <ilqgames/cost/extreme_value_cost.h>
#include <ilqgames/cost/proximity_cost.h>
#include <ilqgames/cost/quadratic_cost.h>
#include <ilqgames/cost/quadratic_polyline2_cost.h>
#include <ilqgames/dynamics/concatenated_dynamical_system.h>
#include <ilqgames/dynamics/single_player_car_5d.h>
#include <ilqgames/examples/skeleton_example.h>
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
// Step 1. Try changing these.
static constexpr float kOmegaCostWeight = 25.0;
static constexpr float kACostWeight = 15.0;
static constexpr float kNominalVCostWeight = 10.0;
static constexpr float kLaneCostWeight = 25.0;
static constexpr float kProximityCostWeight = 100.0;

// Nominal speed.
// Step 2. Try changing these.
static constexpr float kP1NominalV = 8.0;  // m/s
static constexpr float kP2NominalV = 8.0;  // m/s

// Initial state.
static constexpr float kP1InitialX = 0.0;    // m
static constexpr float kP1InitialY = -30.0;  // m

static constexpr float kP2InitialX = -5.0;  // m
static constexpr float kP2InitialY = 30.0;  // m

static constexpr float kP1InitialTheta = M_PI_2;   // rad
static constexpr float kP2InitialTheta = -M_PI_2;  // rad

static constexpr float kP1InitialV = 4.0;  // m/s
static constexpr float kP2InitialV = 3.0;  // m/s

// State dimensions.
using P1 = SinglePlayerCar5D;
using P2 = SinglePlayerCar5D;

static const Dimension kP1XIdx = P1::kPxIdx;
static const Dimension kP1YIdx = P1::kPyIdx;
static const Dimension kP1ThetaIdx = P1::kThetaIdx;
static const Dimension kP1VIdx = P1::kVIdx;

static const Dimension kP2XIdx = P1::kNumXDims + P2::kPxIdx;
static const Dimension kP2YIdx = P1::kNumXDims + P2::kPyIdx;
static const Dimension kP2ThetaIdx = P1::kNumXDims + P2::kThetaIdx;
static const Dimension kP2VIdx = P1::kNumXDims + P2::kVIdx;

}  // anonymous namespace

void SkeletonExample::ConstructDynamics() {
  // Create dynamics. In this case, we have two cars with decoupled dynamics
  // (they are only coupled through the cost structure of the game). This is
  // expressed in the ConcatenatedDynamicalSystem class. Here, each player's
  // dynamics follow that of a standard 5D bicycle model with inter-axle
  // distance below.
  static constexpr float kInterAxleDistance = 4.0;  // m
  dynamics_.reset(new ConcatenatedDynamicalSystem(
      {std::make_shared<P1>(kInterAxleDistance),
       std::make_shared<P2>(kInterAxleDistance)}));
}

void SkeletonExample::ConstructInitialState() {
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
}

void SkeletonExample::ConstructPlayerCosts() {
  // Set up costs for all players. These are containers for holding each
  // player's constituent cost functions and constraints that hold pointwise in
  // time and can apply to either state or control (for *any* player).
  // These costs can also build in regularization on the state or the control,
  // which essentially boils down to adding a scaled identity matrix to each's
  // Hessian.
  player_costs_.emplace_back("P1");
  player_costs_.emplace_back("P2");
  auto& p1_cost = player_costs_[0];
  auto& p2_cost = player_costs_[1];

  // Quadratic control costs.
  const auto p1_omega_cost = std::make_shared<QuadraticCost>(
      kOmegaCostWeight, P1::kOmegaIdx, 0.0, "OmegaCost");
  const auto p1_a_cost = std::make_shared<QuadraticCost>(
      kACostWeight, P1::kAIdx, 0.0, "AccelerationCost");
  p1_cost.AddControlCost(0, p1_omega_cost);
  p1_cost.AddControlCost(0, p1_a_cost);

  const auto p2_omega_cost = std::make_shared<QuadraticCost>(
      kOmegaCostWeight, P2::kOmegaIdx, 0.0, "OmegaCost");
  const auto p2_a_cost = std::make_shared<QuadraticCost>(
      kACostWeight, P2::kAIdx, 0.0, "AccelerationCost");
  p2_cost.AddControlCost(1, p2_omega_cost);
  p2_cost.AddControlCost(1, p2_a_cost);

  // Constrain each control input to lie in an interval.
  // Step 3. Try uncommenting these blocks.
  // const auto p1_omega_max_constraint =
  //     std::make_shared<SingleDimensionConstraint>(
  //         P1::kOmegaIdx, kOmegaMax, true,  "Omega Constraint
  //         (Max)");
  // const auto p1_omega_min_constraint =
  //     std::make_shared<SingleDimensionConstraint>(
  //         P1::kOmegaIdx, -kOmegaMax, false, "Omega Constraint
  //         (Min)");
  // const auto p1_a_max_constraint =
  // std::make_shared<SingleDimensionConstraint>(
  //     P1::kAIdx, kAMax, true, "Acceleration Constraint (Max)");
  // const auto p1_a_min_constraint =
  // std::make_shared<SingleDimensionConstraint>(
  //     P1::kAIdx, -kAMax, false, "Acceleration Constraint
  //     (Min)");
  // p1_cost.AddControlConstraint(0, p1_omega_max_constraint);
  // p1_cost.AddControlConstraint(0, p1_omega_min_constraint);
  // p1_cost.AddControlConstraint(0, p1_a_max_constraint);
  // p1_cost.AddControlConstraint(0, p1_a_min_constraint);

  // const auto p2_omega_max_constraint =
  //     std::make_shared<SingleDimensionConstraint>(
  //         P2::kOmegaIdx, kOmegaMax, true, "Omega Constraint
  //         (Max)");
  // const auto p2_omega_min_constraint =
  //     std::make_shared<SingleDimensionConstraint>(
  //         P2::kOmegaIdx, -kOmegaMax, false, "Omega Constraint
  //         (Min)");
  // const auto p2_a_max_constraint =
  // std::make_shared<SingleDimensionConstraint>(
  //     P2::kAIdx, kAMax, true, "Acceleration Constraint (Max)");
  // const auto p2_a_min_constraint =
  // std::make_shared<SingleDimensionConstraint>(
  //     P2::kAIdx, -kAMax, false, "Acceleration Constraint
  //     (Min)");
  // p2_cost.AddControlConstraint(1, p2_omega_max_constraint);
  // p2_cost.AddControlConstraint(1, p2_omega_min_constraint);
  // p2_cost.AddControlConstraint(1, p2_a_max_constraint);
  // p2_cost.AddControlConstraint(1, p2_a_min_constraint);

  // Encourage each player to go a given nominal speed.
  const auto p1_nominal_v_cost = std::make_shared<QuadraticCost>(
      kNominalVCostWeight, kP1VIdx, kP1NominalV, "NominalV");
  p1_cost.AddStateCost(p1_nominal_v_cost);

  const auto p2_nominal_v_cost = std::make_shared<QuadraticCost>(
      kNominalVCostWeight, kP2VIdx, kP2NominalV, "NominalV");
  p2_cost.AddStateCost(p2_nominal_v_cost);

  // Encourage each player to remain near the lane center. Could also add
  // constraints to stay in the lane.
  const Polyline2 lane1(
      {Point2(kP1InitialX, -1000.0), Point2(kP1InitialX, 1000.0)});
  const Polyline2 lane2({Point2(kP2InitialX, 1000.0), Point2(kP2InitialX, 5.0),
                         Point2(kP2InitialX + 5.0, 0.0),
                         Point2(kP2InitialX + 1000.0, 0.0)});

  const std::shared_ptr<QuadraticPolyline2Cost> p1_lane_cost(
      new QuadraticPolyline2Cost(kLaneCostWeight, lane1, {kP1XIdx, kP1YIdx},
                                 "LaneCenter"));
  p1_cost.AddStateCost(p1_lane_cost);

  const std::shared_ptr<QuadraticPolyline2Cost> p2_lane_cost(
      new QuadraticPolyline2Cost(kLaneCostWeight, lane2, {kP2XIdx, kP2YIdx},
                                 "LaneCenter"));
  p2_cost.AddStateCost(p2_lane_cost);

  // Penalize proximity (could also use a constraint).
  constexpr float kMinProximity = 6.0;  // m
  const std::shared_ptr<ProximityCost> p1p2_proximity_cost(
      new ProximityCost(kProximityCostWeight, {kP1XIdx, kP1YIdx},
                        {kP2XIdx, kP2YIdx}, kMinProximity, "Proximity"));
  p1_cost.AddStateCost(p1p2_proximity_cost);
  p2_cost.AddStateCost(p1p2_proximity_cost);
}

inline std::vector<float> SkeletonExample::Xs(const VectorXf& x) const {
  return {x(kP1XIdx), x(kP2XIdx)};
}

inline std::vector<float> SkeletonExample::Ys(const VectorXf& x) const {
  return {x(kP1YIdx), x(kP2YIdx)};
}

inline std::vector<float> SkeletonExample::Thetas(const VectorXf& x) const {
  return {x(kP1ThetaIdx), x(kP2ThetaIdx)};
}

}  // namespace ilqgames
