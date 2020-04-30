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
// Base class for all iterative LQ game solvers. Derives from FeedbackSolver.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_SOLVER_ILQ_SOLVER_H
#define ILQGAMES_SOLVER_ILQ_SOLVER_H

#include <ilqgames/cost/player_cost.h>
#include <ilqgames/dynamics/multi_player_dynamical_system.h>
#include <ilqgames/solver/game_solver.h>
#include <ilqgames/solver/lq_solver.h>
#include <ilqgames/solver/solver_params.h>
#include <ilqgames/utils/linear_dynamics_approximation.h>
#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/quadratic_cost_approximation.h>
#include <ilqgames/utils/solver_log.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <limits>
#include <memory>
#include <vector>

namespace ilqgames {

class ILQSolver : public GameSolver {
 public:
  virtual ~ILQSolver() {}
  ILQSolver(const std::shared_ptr<const MultiPlayerDynamicalSystem>& dynamics,
            const std::vector<PlayerCost>& player_costs, Time time_horizon,
            LQSolver* lq_solver = nullptr,
            const SolverParams& params = SolverParams())
      : GameSolver(dynamics, player_costs, time_horizon, lq_solver, params) {}

 protected:
  // Populate the given vector with a linearization of the dynamics about
  // the given operating point.
  virtual void ComputeLinearization(
      const OperatingPoint& op,
      std::vector<LinearDynamicsApproximation>* linearization);
};  // class ILQSolver

}  // namespace ilqgames

#endif
