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
// Two player reachability example. Protagonist choosing control to minimize
// max distance (-ve) signed distance to a wall, and antagonist choosing
// disturbance to maximize max signed distance.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/constraint/single_dimension_constraint.h>
#include <ilqgames/cost/polyline2_signed_distance_cost.h>
#include <ilqgames/cost/quadratic_cost.h>
#include <ilqgames/dynamics/two_player_unicycle_4d.h>
#include <ilqgames/examples/two_player_reachability_example.h>
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
DEFINE_double(py0, -10.0, "Initial y-position (m).");
DEFINE_double(theta0, M_PI / 4.0, "Initial heading (rad).");
DEFINE_double(v0, 5.0, "Initial speed (m/s).");

namespace ilqgames {

namespace {

static constexpr float kOmegaMax = 0.1;  // rad/s
static constexpr float kAMax = 1.0;      // m/s/s
static constexpr float kDMax = 0.5;      // m/s
static constexpr float kControlCostWeight = 0.1;

// State dimensions.
using Dyn = TwoPlayerUnicycle4D;

}  // anonymous namespace

void TwoPlayerReachabilityExample::ConstructDynamics() {
  dynamics_.reset(new TwoPlayerUnicycle4D());
}

void TwoPlayerReachabilityExample::ConstructInitialState() {
  x0_ = VectorXf::Zero(dynamics_->XDim());
  x0_(Dyn::kPxIdx) = FLAGS_px0;
  x0_(Dyn::kPyIdx) = FLAGS_py0;
  x0_(Dyn::kThetaIdx) = FLAGS_theta0;
  x0_(Dyn::kVIdx) = FLAGS_v0;
}

void TwoPlayerReachabilityExample::ConstructPlayerCosts() {
  // Set up costs for all players.
  player_costs_.emplace_back("P1");
  player_costs_.emplace_back("P2");
  auto& p1_cost = player_costs_[0];
  auto& p2_cost = player_costs_[1];

  const auto control_cost = std::make_shared<QuadraticCost>(
      kControlCostWeight, -1, 0.0, "ControlCost");
  p1_cost.AddControlCost(0, control_cost);
  p2_cost.AddControlCost(1, control_cost);

  // Target cost.
  static constexpr bool kReach = true;
  const float kTargetRadius = 1.0;
  const Polyline2 circle = DrawCircle(Point2::Zero(), kTargetRadius, 10);
  const std::shared_ptr<Polyline2SignedDistanceCost> p1_target_cost(
      new Polyline2SignedDistanceCost(circle, {Dyn::kPxIdx, Dyn::kPyIdx},
                                      !kReach, "Target"));
  const std::shared_ptr<Polyline2SignedDistanceCost> p2_target_cost(
      new Polyline2SignedDistanceCost(circle, {Dyn::kPxIdx, Dyn::kPyIdx},
                                      kReach, "Target"));

  p1_cost.AddStateCost(p1_target_cost);
  p2_cost.AddStateCost(p2_target_cost);

  // Make sure costs are max-over-time.
  p1_cost.SetMaxOverTime();
  p2_cost.SetMinOverTime();
}

inline std::vector<float> TwoPlayerReachabilityExample::Xs(
    const VectorXf& x) const {
  return {x(Dyn::kPxIdx)};
}

inline std::vector<float> TwoPlayerReachabilityExample::Ys(
    const VectorXf& x) const {
  return {x(Dyn::kPyIdx)};
}

inline std::vector<float> TwoPlayerReachabilityExample::Thetas(
    const VectorXf& x) const {
  return {x(Dyn::kThetaIdx)};
}

}  // namespace ilqgames
