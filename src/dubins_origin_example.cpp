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
// Two Dubins cars. One tries to follow other one, which is trying to get to the
// origin. Both also have penalties on control input, and we are interested in
// the difference between open-loop and feedback Nash strategies for this
// situation. Example conceived by Forrest Laine ( forrest.laine@berkeley.edu ).
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/cost/quadratic_cost.h>
#include <ilqgames/cost/quadratic_difference_cost.h>
#include <ilqgames/dynamics/concatenated_dynamical_system.h>
#include <ilqgames/dynamics/single_player_dubins_car.h>
#include <ilqgames/examples/dubins_origin_example.h>
#include <ilqgames/solver/ilq_solver.h>
#include <ilqgames/solver/problem.h>
#include <ilqgames/solver/solver_params.h>
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

// Cost weights.
static constexpr float kOmegaCostWeight = 100.0;
static constexpr float kAttractionCostWeight = 10.0;
static constexpr float kGoalCostWeight = 10.0;

// Initial state.
static constexpr float kP1InitialX = 10.0;               // m
static constexpr float kP1InitialY = -10.0;              // m
static constexpr float kP1InitialHeading = 0.75 * M_PI;  // rad

static constexpr float kP2InitialX = 10.0;               // m
static constexpr float kP2InitialY = 10.0;               // m
static constexpr float kP2InitialHeading = 1.25 * M_PI;  // rad

static constexpr float kSpeed = 1.0;  // m/s

// Goal position.
static constexpr float kP1GoalX = 0.0;
static constexpr float kP1GoalY = 0.0;

// State dimensions.
using P1 = SinglePlayerDubinsCar;
using P2 = SinglePlayerDubinsCar;

static const Dimension kP1XIdx = P1::kPxIdx;
static const Dimension kP1YIdx = P1::kPyIdx;
static const Dimension kP1HeadingIdx = P1::kThetaIdx;

static const Dimension kP2XIdx = P1::kNumXDims + P2::kPxIdx;
static const Dimension kP2YIdx = P1::kNumXDims + P2::kPyIdx;
static const Dimension kP2HeadingIdx = P1::kNumXDims + P2::kThetaIdx;

// Control dimensions.
static const Dimension kP1OmegaIdx = 0;
static const Dimension kP2OmegaIdx = 0;
}  // anonymous namespace

DubinsOriginExample::DubinsOriginExample(const SolverParams& params) {
  // Create dynamics.
  const std::shared_ptr<const ConcatenatedDynamicalSystem> dynamics(
      new ConcatenatedDynamicalSystem(
          {std::make_shared<P1>(kSpeed), std::make_shared<P2>(kSpeed)},
          kTimeStep));

  // Set up initial state.
  x0_ = VectorXf::Zero(dynamics->XDim());
  x0_(kP1XIdx) = kP1InitialX;
  x0_(kP1YIdx) = kP1InitialY;
  x0_(kP1HeadingIdx) = kP1InitialHeading;
  x0_(kP2XIdx) = kP2InitialX;
  x0_(kP2YIdx) = kP2InitialY;
  x0_(kP2HeadingIdx) = kP2InitialHeading;

  // Set up initial strategies and operating point.
  strategies_.reset(new std::vector<Strategy>());
  for (PlayerIndex ii = 0; ii < dynamics->NumPlayers(); ii++)
    strategies_->emplace_back(kNumTimeSteps, dynamics->XDim(),
                              dynamics->UDim(ii));

  operating_point_.reset(
      new OperatingPoint(kNumTimeSteps, dynamics->NumPlayers(), 0.0, dynamics));

  // Set up costs for all players.
  PlayerCost p1_cost("P1"), p2_cost("P2");

  // Attract P2 to P1.
  const std::shared_ptr<QuadraticDifferenceCost> p2_attraction_cost =
      std::shared_ptr<QuadraticDifferenceCost>(
          new QuadraticDifferenceCost(kAttractionCostWeight, {kP1XIdx, kP1YIdx},
                                      {kP2XIdx, kP2YIdx}, "AttractionCost"));
  p2_cost.AddStateCost(p2_attraction_cost);

  // Penalize control effort.
  const auto p1_omega_cost = std::make_shared<QuadraticCost>(
      kOmegaCostWeight, kP1OmegaIdx, 0.0, "Steering");
  p1_cost.AddControlCost(0, p1_omega_cost);

  const auto p2_omega_cost = std::make_shared<QuadraticCost>(
      kOmegaCostWeight, kP2OmegaIdx, 0.0, "Steering");
  p2_cost.AddControlCost(1, p2_omega_cost);

  // Goal cost.
  const auto p1_goalx_cost = std::make_shared<QuadraticCost>(
      kGoalCostWeight, kP1XIdx, kP1GoalX, "GoalX");
  const auto p1_goaly_cost = std::make_shared<QuadraticCost>(
      kGoalCostWeight, kP1YIdx, kP1GoalY, "GoalY");
  p1_cost.AddStateCost(p1_goalx_cost);
  p1_cost.AddStateCost(p1_goaly_cost);

  // Set up solver.
  solver_.reset(
      new ILQSolver(dynamics, {p1_cost, p2_cost}, kTimeHorizon, params));
}

inline std::vector<float> DubinsOriginExample::Xs(const VectorXf& x) const {
  return {x(kP1XIdx), x(kP2XIdx)};
}

inline std::vector<float> DubinsOriginExample::Ys(const VectorXf& x) const {
  return {x(kP1YIdx), x(kP2YIdx)};
}

inline std::vector<float> DubinsOriginExample::Thetas(const VectorXf& x) const {
  return {x(kP1HeadingIdx), x(kP2HeadingIdx)};
}

}  // namespace ilqgames