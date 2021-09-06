/*
 * Copyright (c) 2021, The Regents of the University of California (Regents).
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
 * Authors: David Fridovich-Keil   ( dfk@utexas.edu )
 *          Jaime Fisac            ( jfisac@princeton.edu )
 */

///////////////////////////////////////////////////////////////////////////////
//
// One player reach-avoid example. Single player choosing control reach target
// ball while staying clear of failure balls.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/constraint/single_dimension_constraint.h>
#include <ilqgames/cost/polyline2_signed_distance_cost.h>
#include <ilqgames/cost/quadratic_cost.h>
#include <ilqgames/dynamics/concatenated_dynamical_system.h>
#include <ilqgames/dynamics/single_player_car_5d.h>
#include <ilqgames/dynamics/single_player_dubins_car.h>
#include <ilqgames/examples/one_player_reach_avoid_example.h>
#include <ilqgames/geometry/draw_shapes.h>
#include <ilqgames/geometry/polyline2.h>
#include <ilqgames/solver/ilq_solver.h>
#include <ilqgames/solver/problem.h>
#include <ilqgames/solver/solver_params.h>
#include <ilqgames/utils/types.h>

#include <gflags/gflags.h>
#include <math.h>
#include <memory>
#include <vector>

// Initial state command-line flags.
DEFINE_double(px0, 0.0, "Initial x-position (m).");
DEFINE_double(py0, 0.0, "Initial y-position (m).");
DEFINE_double(theta0, 0.0, "Initial heading (rad).");
DEFINE_double(v0, 1.0, "Initial speed (m/s).");

namespace ilqgames {

namespace {

// Target/failure radii.
static constexpr float kTargetRadius = 2.0;   // m
static constexpr float kFailureRadius = 2.0;  // m

// Input cost weight.
static constexpr float kControlCostWeight = 0.1;

// Speed.
static constexpr float kInterAxleDistance = 4.0;  // m

// Target position.
static constexpr float kP1TargetX = 15.0;
static constexpr float kP1TargetY = 0.0;

// Obstacle position.
static constexpr float kP1FailureX = 10.0;
static constexpr float kP1FailureY = 0.0;

// State dimensions.
using P1 = SinglePlayerCar5D;

static const Dimension kP1XIdx = P1::kPxIdx;
static const Dimension kP1YIdx = P1::kPyIdx;
static const Dimension kP1ThetaIdx = P1::kThetaIdx;
static const Dimension kP1PhiIdx = P1::kPhiIdx;
static const Dimension kP1VIdx = P1::kVIdx;

// Control dimensions.
static const Dimension kP1OmegaIdx = P1::kOmegaIdx;
static const Dimension kP1AIdx = P1::kAIdx;

}  // anonymous namespace

void OnePlayerReachAvoidExample::ConstructDynamics() {
  dynamics_.reset(new ConcatenatedDynamicalSystem(
      {std::make_shared<P1>(kInterAxleDistance)}));
}

void OnePlayerReachAvoidExample::ConstructInitialState() {
  // Set up initial state.
  x0_ = VectorXf::Zero(dynamics_->XDim());
  x0_(kP1XIdx) = FLAGS_px0;
  x0_(kP1YIdx) = FLAGS_py0;
  x0_(kP1ThetaIdx) = FLAGS_theta0;
  x0_(kP1VIdx) = FLAGS_v0;
}

void OnePlayerReachAvoidExample::ConstructPlayerCosts() {
  // Set up costs for all players.
  player_costs_.emplace_back("P1");
  auto& p1_cost = player_costs_[0];

  const auto control_cost = std::make_shared<QuadraticCost>(
      kControlCostWeight, -1, 0.0, "ControlCost");
  p1_cost.AddControlCost(0, control_cost);

  // // Constrain control effort.
  // const auto p1_omega_max_constraint =
  //     std::make_shared<SingleDimensionConstraint>(kP1OmegaIdx, kOmegaMax,
  //     true,
  //                                                 "Input Constraint (Max)");
  // const auto p1_omega_min_constraint =
  //     std::make_shared<SingleDimensionConstraint>(
  //         kP1OmegaIdx, -kOmegaMax, false, "Input Constraint (Min)");
  // p1_cost.AddControlConstraint(0, p1_omega_max_constraint);
  // p1_cost.AddControlConstraint(0, p1_omega_min_constraint);

  // Target cost.
  const Polyline2 target =
      DrawCircle(Point2(kP1TargetX, kP1TargetY), kTargetRadius, 10);
  const std::shared_ptr<Polyline2SignedDistanceCost> p1_target_cost(
      new Polyline2SignedDistanceCost(target, {kP1XIdx, kP1YIdx}, 0.0, true));
  p1_cost.SetTargetStateCost(std::shared_ptr<ExtremeValueCost>(
      new ExtremeValueCost({p1_target_cost}, true, "Target")));

  // Failure cost.
  const Polyline2 failure =
      DrawCircle(Point2(kP1TargetX, kP1TargetY), kTargetRadius, 10);
  const std::shared_ptr<Polyline2SignedDistanceCost> p1_failure_cost(
      new Polyline2SignedDistanceCost(failure, {kP1XIdx, kP1YIdx}, 0.0, false));
  p1_cost.SetTargetStateCost(std::shared_ptr<ExtremeValueCost>(
      new ExtremeValueCost({p1_failure_cost}, false, "Failure")));

  // Make sure costs are reach-avoid.
  p1_cost.SetReachAvoid();
}

inline std::vector<float> OnePlayerReachAvoidExample::Xs(
    const VectorXf& x) const {
  return {x(kP1XIdx)};
}

inline std::vector<float> OnePlayerReachAvoidExample::Ys(
    const VectorXf& x) const {
  return {x(kP1YIdx)};
}

inline std::vector<float> OnePlayerReachAvoidExample::Thetas(
    const VectorXf& x) const {
  return {x(kP1ThetaIdx)};
}

}  // namespace ilqgames
