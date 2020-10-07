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
// Two player Air3D example from:
// https://www.cs.ubc.ca/~mitchell/Papers/publishedIEEEtac05.pdf.
//
// Modified such that it does *not* use relative dynamics and instead considers
// the motion of each player separately.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/constraint/single_dimension_constraint.h>
#include <ilqgames/cost/quadratic_cost.h>
#include <ilqgames/cost/quadratic_difference_cost.h>
#include <ilqgames/cost/relative_distance_cost.h>
#include <ilqgames/dynamics/concatenated_dynamical_system.h>
#include <ilqgames/dynamics/single_player_dubins_car.h>
#include <ilqgames/dynamics/single_player_point_mass_2d.h>
#include <ilqgames/examples/modified_air_3d_example.h>
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
DEFINE_double(rx0, 4.0, "Initial x-position (m).");
DEFINE_double(ry0, 3.0, "Initial y-position (m).");
DEFINE_double(rtheta0, M_PI / 4.0, "Initial heading (rad).");
DEFINE_double(ve, 1.0, "Evader speed (m/s).");
DEFINE_double(vp, 1.0, "Pursuer speed (m/s).");

namespace ilqgames {

namespace {

// Input constraint and cost weight.
static constexpr float kOmegaMax = 1.0;  // rad/s
static constexpr float kOmegaCostWeight = 0.1;

// State dimensions.
// using Dyn = SinglePlayerDubinsCar;
using Dyn = SinglePlayerPointMass2D;

static const Dimension kP1PxIdx = Dyn::kPxIdx;
static const Dimension kP1PyIdx = Dyn::kPyIdx;
// static const Dimension kP1ThetaIdx = Dyn::kThetaIdx;
static const Dimension kP1VxIdx = Dyn::kVxIdx;
static const Dimension kP1VyIdx = Dyn::kVyIdx;

static const Dimension kP2PxIdx = Dyn::kNumXDims + Dyn::kPxIdx;
static const Dimension kP2PyIdx = Dyn::kNumXDims + Dyn::kPyIdx;
// static const Dimension kP2ThetaIdx = Dyn::kNumXDims + Dyn::kThetaIdx;
static const Dimension kP2VxIdx = Dyn::kNumXDims + Dyn::kVxIdx;
static const Dimension kP2VyIdx = Dyn::kNumXDims + Dyn::kVyIdx;

}  // anonymous namespace

void ModifiedAir3DExample::ConstructDynamics() {
  dynamics_.reset(new ConcatenatedDynamicalSystem(
      {std::make_shared<Dyn>(), std::make_shared<Dyn>()}));
}

void ModifiedAir3DExample::ConstructInitialState() {
  // Set up initial state.
  x0_ = VectorXf::Zero(dynamics_->XDim());
  x0_(kP1VxIdx) = FLAGS_ve;
  x0_(kP2PxIdx) = FLAGS_rx0;
  x0_(kP2PyIdx) = FLAGS_ry0;
  x0_(kP2VxIdx) = FLAGS_vp * std::cos(FLAGS_rtheta0);
  x0_(kP2VyIdx) = FLAGS_vp * std::sin(FLAGS_rtheta0);
}

void ModifiedAir3DExample::ConstructPlayerCosts() {
  // Set up costs for all players.
  player_costs_.emplace_back("P1", 1.0, 0.0);
  player_costs_.emplace_back("P2", 1.0, 0.0);
  auto& p1_cost = player_costs_[0];
  auto& p2_cost = player_costs_[1];

  const auto control_cost =
      std::make_shared<QuadraticCost>(kOmegaCostWeight, -1, 0.0, "ControlCost");
  p1_cost.AddControlCost(0, control_cost);
  p2_cost.AddControlCost(1, control_cost);

  // Constrain control effort.
  // const auto omega_max_constraint =
  // std::make_shared<SingleDimensionConstraint>(
  //     Dyn::kOmegaIdx, kOmegaMax, true, "Omega Constraint
  //     (Max)");
  // const auto omega_min_constraint =
  // std::make_shared<SingleDimensionConstraint>(
  //     Dyn::kOmegaIdx, -kOmegaMax, false, "Omega Constraint
  //     (Min)");
  // p1_cost.AddControlConstraint(0, omega_max_constraint);
  // p1_cost.AddControlConstraint(0, omega_min_constraint);
  // p2_cost.AddControlConstraint(1, omega_max_constraint);
  // p2_cost.AddControlConstraint(1, omega_min_constraint);

  // Target cost.
  constexpr float kEvaderWeight = -1e6;
  constexpr float kPursuerWeight = 1e6;
  // const std::shared_ptr<RelativeDistanceCost> p1_target_cost(
  //     new RelativeDistanceCost(kEvaderWeight, {kP1PxIdx, kP1PyIdx},
  //                              {kP2PxIdx, kP2PyIdx}, "Target"));
  // const std::shared_ptr<RelativeDistanceCost> p2_target_cost(
  //     new RelativeDistanceCost(kPursuerWeight, {kP1PxIdx, kP1PyIdx},
  //                              {kP2PxIdx, kP2PyIdx}, "Target"));
  const std::shared_ptr<QuadraticDifferenceCost> p1_target_cost(
      new QuadraticDifferenceCost(kEvaderWeight, {kP1PxIdx, kP1PyIdx},
                                  {kP2PxIdx, kP2PyIdx}, "Target"));
  const std::shared_ptr<QuadraticDifferenceCost> p2_target_cost(
      new QuadraticDifferenceCost(kPursuerWeight, {kP1PxIdx, kP1PyIdx},
                                  {kP2PxIdx, kP2PyIdx}, "Target"));
  p1_cost.AddStateCost(p1_target_cost);
  p2_cost.AddStateCost(p2_target_cost);

  // Make sure evader's cost is a max-over-time and pursuer's is a
  // min-over-time.
  // p1_cost.SetMaxOverTime();
  // p2_cost.SetMinOverTime();
}

inline std::vector<float> ModifiedAir3DExample::Xs(const VectorXf& x) const {
  return {x(kP1PxIdx), x(kP2PxIdx)};
}

inline std::vector<float> ModifiedAir3DExample::Ys(const VectorXf& x) const {
  return {x(kP1PyIdx), x(kP2PyIdx)};
}

inline std::vector<float> ModifiedAir3DExample::Thetas(
    const VectorXf& x) const {
  //  return {x(kP1ThetaIdx), x(kP2ThetaIdx)};
  return {std::atan2(x(kP1VyIdx), x(kP1VxIdx)),
          std::atan2(x(kP2VyIdx), x(kP2VxIdx))};
}

}  // namespace ilqgames
