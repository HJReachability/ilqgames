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

#ifndef ILQGAMES_SOLVER_LINESEARCHING_ILQ_SOLVER_H
#define ILQGAMES_SOLVER_LINESEARCHING_ILQ_SOLVER_H

#include <ilqgames/cost/player_cost.h>
#include <ilqgames/dynamics/multi_player_dynamical_system.h>
#include <ilqgames/solver/ilq_solver.h>
#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/quadratic_cost_approximation.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

#include <memory>
#include <vector>

namespace ilqgames {

class LinesearchingILQSolver : public ILQSolver {
 public:
  virtual ~LinesearchingILQSolver() {}
  LinesearchingILQSolver(
      const std::shared_ptr<const MultiPlayerDynamicalSystem>& dynamics,
      const std::vector<PlayerCost>& player_costs, Time time_horizon,
      Time time_step)
      : ILQSolver(dynamics, player_costs, time_horizon, time_step) {}

 protected:
  // Modify LQ strategies to improve convergence properties.
  // This function replaces an Armijo linesearch that would take place in ILQR.
  // Returns true if successful.
  bool ModifyLQStrategies(const OperatingPoint& current_operating_point,
                          std::vector<Strategy>* strategies) const;
};  // class LinesearchingILQSolver

}  // namespace ilqgames

#endif
