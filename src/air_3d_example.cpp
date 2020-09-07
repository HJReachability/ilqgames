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
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/constraint/single_dimension_constraint.h>
#include <ilqgames/cost/polyline2_signed_distance_cost.h>
#include <ilqgames/cost/quadratic_cost.h>
#include <ilqgames/dynamics/air_3d.h>
#include <ilqgames/examples/air_3d_example.h>
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
// Time.
static constexpr Time kTimeStep = 0.1;      // s
static constexpr Time kTimeHorizon = 10.0;  // s
static constexpr size_t kNumTimeSteps =
    static_cast<size_t>(kTimeHorizon / kTimeStep);

// Input constraint.
static constexpr float kOmegaMax = 1.0;  // rad/s

// State dimensions.
using Dyn = Air3D;
}  // anonymous namespace

Air3DExample::Air3DExample(const SolverParams& params) {
  // Create dynamics.
  const auto dynamics =
      std::make_shared<const Air3D>(FLAGS_ve, FLAGS_vp, kTimeStep);

  // Set up initial state.
  x0_ = VectorXf::Zero(dynamics->XDim());
  x0_(Dyn::kRxIdx) = FLAGS_rx0;
  x0_(Dyn::kRyIdx) = FLAGS_ry0;
  x0_(Dyn::kRThetaIdx) = FLAGS_rtheta0;

  // Set up initial strategies and operating point.
  strategies_.reset(new std::vector<Strategy>());
  for (PlayerIndex ii = 0; ii < dynamics->NumPlayers(); ii++)
    strategies_->emplace_back(kNumTimeSteps, dynamics->XDim(),
                              dynamics->UDim(ii));

  operating_point_.reset(
      new OperatingPoint(kNumTimeSteps, dynamics->NumPlayers(), 0.0, dynamics));

  // Set up costs for all players.
  PlayerCost p1_cost("P1"), p2_cost("P2");

  const auto control_cost = std::make_shared<QuadraticCost>(
      params.control_regularization, -1, 0.0, "ControlCost");
  p1_cost.AddControlCost(0, control_cost);
  p2_cost.AddControlCost(1, control_cost);

  // Constrain control effort.
  const auto p1_omega_max_constraint =
      std::make_shared<SingleDimensionConstraint>(
          Dyn::kOmega1Idx, kOmegaMax, false, "Omega Constraint (Max)");
  const auto p1_omega_min_constraint =
      std::make_shared<SingleDimensionConstraint>(
          Dyn::kOmega1Idx, -kOmegaMax, true, "Omega Constraint (Min)");
   p1_cost.AddControlConstraint(0, p1_omega_max_constraint);
   p1_cost.AddControlConstraint(0, p1_omega_min_constraint);

  const auto p2_omega_max_constraint =
      std::make_shared<SingleDimensionConstraint>(
          Dyn::kOmega1Idx, kOmegaMax, false, "Omega Constraint (Max)");
  const auto p2_omega_min_constraint =
      std::make_shared<SingleDimensionConstraint>(
          Dyn::kOmega2Idx, -kOmegaMax, true, "Omega Constraint (Min)");
   p2_cost.AddControlConstraint(1, p2_omega_max_constraint);
   p2_cost.AddControlConstraint(1, p2_omega_min_constraint);

  // Target cost.
  const float kTargetRadius = 5.0;
  const Polyline2 circle = DrawCircle(Point2::Zero(), kTargetRadius, 10);

  constexpr bool kReach = true;
  const std::shared_ptr<Polyline2SignedDistanceCost> p1_target_cost(
      new Polyline2SignedDistanceCost(circle, {Dyn::kRxIdx, Dyn::kRyIdx},
                                      !kReach, "Target"));
  const std::shared_ptr<Polyline2SignedDistanceCost> p2_target_cost(
      new Polyline2SignedDistanceCost(circle, {Dyn::kRxIdx, Dyn::kRyIdx},
                                      kReach, "Target"));

  p1_cost.AddStateCost(p1_target_cost);
  p2_cost.AddStateCost(p2_target_cost);

  // Make sure evader's cost is a max-over-time and pursuer's is a
  // min-over-time.
  p1_cost.SetMaxOverTime();
  p2_cost.SetMinOverTime();

  // Set up solver.
  solver_.reset(
      new ILQSolver(dynamics, {p1_cost, p2_cost}, kTimeHorizon, params));
}

inline std::vector<float> Air3DExample::Xs(const VectorXf& x) const {
  return {0.0, x(Dyn::kRxIdx)};
}

inline std::vector<float> Air3DExample::Ys(const VectorXf& x) const {
  return {0.0, x(Dyn::kRyIdx)};
}

inline std::vector<float> Air3DExample::Thetas(const VectorXf& x) const {
  return {0.0, x(Dyn::kRThetaIdx)};
}

}  // namespace ilqgames
