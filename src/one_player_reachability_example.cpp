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
// One player reachability example. Single player choosing control to minimize
// max distance (-ve) signed distance to a ball we're outside.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/constraint/single_dimension_constraint.h>
#include <ilqgames/cost/polyline2_signed_distance_cost.h>
#include <ilqgames/cost/quadratic_cost.h>
#include <ilqgames/dynamics/concatenated_dynamical_system.h>
#include <ilqgames/dynamics/single_player_dubins_car.h>
#include <ilqgames/examples/one_player_reachability_example.h>
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
DEFINE_double(px0, 1.75, "Initial x-position (m).");
DEFINE_double(py0, 1.75, "Initial y-position (m).");
DEFINE_double(theta0, 0.0, "Initial heading (rad).");

namespace ilqgames {

namespace {

// Reach or avoid?
static constexpr bool kAvoid = true;

// Target radius.
static constexpr float kTargetRadius = 2.0;  // m

// Input constraint and cost weight.
static constexpr float kOmegaMax = 1.0;  // rad/s
static constexpr float kOmegaCostWeight = 0.1;

// Speed.
static constexpr float kSpeed = 1.0;  // m/s

// Target position.
static constexpr float kP1TargetX = 0.0;
static constexpr float kP1TargetY = 0.0;

// State dimensions.
using P1 = SinglePlayerDubinsCar;

static const Dimension kP1XIdx = P1::kPxIdx;
static const Dimension kP1YIdx = P1::kPyIdx;
static const Dimension kP1ThetaIdx = P1::kThetaIdx;

// Control dimensions.
static const Dimension kP1OmegaIdx = 0;

}  // anonymous namespace

void OnePlayerReachabilityExample::ConstructDynamics() {
  dynamics_.reset(
      new ConcatenatedDynamicalSystem({std::make_shared<P1>(kSpeed)}));
}

void OnePlayerReachabilityExample::ConstructInitialState() {
  // Set up initial state.
  x0_ = VectorXf::Zero(dynamics_->XDim());
  x0_(kP1XIdx) = FLAGS_px0;
  x0_(kP1YIdx) = FLAGS_py0;
  x0_(kP1ThetaIdx) = FLAGS_theta0;
}

void OnePlayerReachabilityExample::ConstructPlayerCosts() {
  // Set up costs for all players.
  player_costs_.emplace_back("P1");
  auto& p1_cost = player_costs_[0];

  const auto control_cost =
      std::make_shared<QuadraticCost>(kOmegaCostWeight, -1, 0.0, "ControlCost");
  p1_cost.AddControlCost(0, control_cost);

  // Constrain control effort.
  const auto p1_omega_max_constraint =
      std::make_shared<SingleDimensionConstraint>(kP1OmegaIdx, kOmegaMax, true,
                                                  "Input Constraint (Max)");
  const auto p1_omega_min_constraint =
      std::make_shared<SingleDimensionConstraint>(
          kP1OmegaIdx, -kOmegaMax, false, "Input Constraint (Min)");
  p1_cost.AddControlConstraint(0, p1_omega_max_constraint);
  p1_cost.AddControlConstraint(0, p1_omega_min_constraint);

  // Target cost.
  const Polyline2 circle =
      DrawCircle(Point2(kP1TargetX, kP1TargetY), kTargetRadius, 10);
  const std::shared_ptr<Polyline2SignedDistanceCost> p1_target_cost(
      new Polyline2SignedDistanceCost(circle, {kP1XIdx, kP1YIdx}, kAvoid,
                                      "Target"));

  p1_cost.AddStateCost(p1_target_cost);

  // Make sure costs are maxima-over-time.
  p1_cost.SetMaxOverTime();
}

inline std::vector<float> OnePlayerReachabilityExample::Xs(
    const VectorXf& x) const {
  return {x(kP1XIdx)};
}

inline std::vector<float> OnePlayerReachabilityExample::Ys(
    const VectorXf& x) const {
  return {x(kP1YIdx)};
}

inline std::vector<float> OnePlayerReachabilityExample::Thetas(
    const VectorXf& x) const {
  return {x(kP1ThetaIdx)};
}

}  // namespace ilqgames
