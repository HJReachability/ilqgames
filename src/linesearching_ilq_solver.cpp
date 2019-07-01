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
// More stable version of the ILQSolver class, in which the method
// 'ModifyLQStrategies' is overridden to implement a simple linesearch based on
// bounding the change in operating points.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/cost/player_cost.h>
#include <ilqgames/dynamics/multi_player_dynamical_system.h>
#include <ilqgames/solver/linesearching_ilq_solver.h>
#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/quadratic_cost_approximation.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

#include <memory>
#include <vector>

namespace ilqgames {

namespace {
// Check if the maximum elementwise state difference between two
// operating points is small (according to a hard-coded constant).
bool AreOperatingPointsClose(const OperatingPoint& op1,
                             const OperatingPoint& op2) {
  CHECK_EQ(op1.xs.size(), op2.xs.size());

  constexpr float kMaxElementwiseDifference = 1.0;
  for (size_t kk = 0; kk < op1.xs.size(); kk++) {
    if ((op1.xs[kk] - op2.xs[kk]).cwiseAbs().maxCoeff() >
        kMaxElementwiseDifference)
      return false;
  }

  return true;
}

// Multiply all alphas in a set of strategies by the given constant.
void ScaleAlphas(float scaling, std::vector<Strategy>* strategies) {
  CHECK_NOTNULL(strategies);

  for (auto& strategy : *strategies) {
    for (auto& alpha : strategy.alphas) alpha *= scaling;
  }
}

}  // anonymous namespace

bool LinesearchingILQSolver::ModifyLQStrategies(
    const OperatingPoint& current_operating_point,
    std::vector<Strategy>* strategies) const {
  // Compute next operating point.
  OperatingPoint next_operating_point(num_time_steps_, dynamics_->NumPlayers(),
                                      current_operating_point.t0);
  CurrentOperatingPoint(current_operating_point, *strategies,
                        &next_operating_point);

  // Initially scale alphas by a fixed amount to avoid unnecessary backtracking.
  constexpr float kInitialAlphaScaling = 0.1;
  //  ScaleAlphas(kInitialAlphaScaling, strategies);

  // Keep halving alphas until the maximum elementwise state difference is above
  // a threshold.
  constexpr float kGeometricAlphaScaling = 0.5;
  constexpr size_t kMaxBacktrackingSteps = 100;
  for (size_t ii = 0; ii < kMaxBacktrackingSteps; ii++) {
    if (AreOperatingPointsClose(current_operating_point, next_operating_point))
      return true;

    ScaleAlphas(kGeometricAlphaScaling, strategies);
    CurrentOperatingPoint(current_operating_point, *strategies,
                          &next_operating_point);
  }

  LOG(WARNING) << "Exceeded maximum number of backtracking steps.";
  return false;
}

}  // namespace ilqgames
