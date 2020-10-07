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
// Two player collision-avoidance example using approximate HJ reachability.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/cost/quadratic_cost.h>
#include <ilqgames/cost/signed_distance_cost.h>
#include <ilqgames/dynamics/concatenated_dynamical_system.h>
#include <ilqgames/dynamics/single_player_car_5d.h>
#include <ilqgames/examples/two_player_collision_avoidance_reachability_example.h>
#include <ilqgames/geometry/draw_shapes.h>
#include <ilqgames/geometry/polyline2.h>
#include <ilqgames/solver/ilq_solver.h>
#include <ilqgames/solver/problem.h>
#include <ilqgames/solver/solver_params.h>
#include <ilqgames/utils/types.h>

#include <math.h>
#include <memory>
#include <vector>

// Initial state command-line flags.
DEFINE_double(px0, 0.0, "Initial x-position (m).");
DEFINE_double(py0, -5.0, "Initial y-position (m).");

namespace ilqgames {

namespace {

// Control cost weight.
static constexpr float kOmegaCostWeight = 0.1;

// Initial state.
static constexpr float kP1InitialHeading = 0.1;  // rad
static constexpr float kP1InitialSpeed = 5.0;    // m/s
static constexpr float kP2InitialX = 0.0;        // m
static constexpr float kP2InitialY = 0.0;        // m
static constexpr float kP2InitialHeading = 0.0;  // rad
static constexpr float kP2InitialSpeed = 5.0;    // m/s

// State dimensions.
using P1 = SinglePlayerCar5D;
using P2 = SinglePlayerCar5D;
static constexpr float kInterAxleDistance = 4.0;

static const Dimension kP1XIdx = P1::kPxIdx;
static const Dimension kP1YIdx = P1::kPyIdx;
static const Dimension kP1HeadingIdx = P1::kThetaIdx;
static const Dimension kP1VIdx = P1::kVIdx;

static const Dimension kP2XIdx = P1::kNumXDims + P2::kPxIdx;
static const Dimension kP2YIdx = P1::kNumXDims + P2::kPyIdx;
static const Dimension kP2HeadingIdx = P1::kNumXDims + P2::kThetaIdx;
static const Dimension kP2VIdx = P1::kNumXDims + P2::kVIdx;

}  // anonymous namespace

void TwoPlayerCollisionAvoidanceReachabilityExample::ConstructDynamics() {
  dynamics_.reset(new ConcatenatedDynamicalSystem(
      {std::make_shared<P1>(kInterAxleDistance),
       std::make_shared<P2>(kInterAxleDistance)}));
}

void TwoPlayerCollisionAvoidanceReachabilityExample::ConstructInitialState() {
  x0_ = VectorXf::Zero(dynamics_->XDim());
  x0_(kP1XIdx) = FLAGS_px0;
  x0_(kP1YIdx) = FLAGS_py0;
  x0_(kP1HeadingIdx) = kP1InitialHeading;
  x0_(kP1VIdx) = kP1InitialSpeed;
  x0_(kP2XIdx) = kP2InitialX;
  x0_(kP2YIdx) = kP2InitialY;
  x0_(kP2HeadingIdx) = kP2InitialHeading;
  x0_(kP2VIdx) = kP2InitialSpeed;
}

void TwoPlayerCollisionAvoidanceReachabilityExample::ConstructPlayerCosts() {
  // Set up costs for all players.
  player_costs_.emplace_back("P1");
  player_costs_.emplace_back("P2");
  auto& p1_cost = player_costs_[0];
  auto& p2_cost = player_costs_[1];

  const auto control_cost =
      std::make_shared<QuadraticCost>(kOmegaCostWeight, -1, 0.0, "ControlCost");
  p1_cost.AddControlCost(0, control_cost);
  p2_cost.AddControlCost(1, control_cost);

  // Collision-avoidance cost.
  auto p1_position = [](Time t) {
    return Point2(FLAGS_px0, FLAGS_py0) +
           t * kP1InitialSpeed *
               Point2(std::cos(kP1InitialHeading), std::sin(kP1InitialHeading));
  };  // p1_position
  auto p2_position = [](Time t) {
    return Point2(kP2InitialX, kP2InitialY) +
           t * kP2InitialSpeed *
               Point2(std::cos(kP2InitialHeading), std::sin(kP2InitialHeading));
  };  // p2_position

  // NOTE: Assumes line segments traced by each player at initialization do not
  // intersect.
  const float nominal_distance = (p1_position(0.5 * time::kTimeHorizon) -
                                  p2_position(0.5 * time::kTimeHorizon))
                                     .norm();
  const std::shared_ptr<SignedDistanceCost> collision_avoidance_cost(
      new SignedDistanceCost({kP1XIdx, kP1YIdx}, {kP2XIdx, kP2YIdx},
                             nominal_distance, "CollisionAvoidance"));
  p1_cost.AddStateCost(collision_avoidance_cost);
  p2_cost.AddStateCost(collision_avoidance_cost);

  // Make sure costs are max-over-time.
  p1_cost.SetMaxOverTime();
  p2_cost.SetMaxOverTime();
}

inline std::vector<float> TwoPlayerCollisionAvoidanceReachabilityExample::Xs(
    const VectorXf& x) const {
  return {x(kP1XIdx), x(kP2XIdx)};
}

inline std::vector<float> TwoPlayerCollisionAvoidanceReachabilityExample::Ys(
    const VectorXf& x) const {
  return {x(kP1YIdx), x(kP2YIdx)};
}

inline std::vector<float>
TwoPlayerCollisionAvoidanceReachabilityExample::Thetas(
    const VectorXf& x) const {
  return {x(kP1HeadingIdx), x(kP2HeadingIdx)};
}

}  // namespace ilqgames
