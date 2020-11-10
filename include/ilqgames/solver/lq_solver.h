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
//  Base class for all LQ game solvers. For further details please refer to
//  derived class comments.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_SOLVER_LQ_SOLVER_H
#define ILQGAMES_SOLVER_LQ_SOLVER_H

#include <ilqgames/dynamics/multi_player_integrable_system.h>
#include <ilqgames/utils/linear_dynamics_approximation.h>
#include <ilqgames/utils/quadratic_cost_approximation.h>
#include <ilqgames/utils/strategy.h>

#include <glog/logging.h>
#include <vector>

namespace ilqgames {

class LQSolver {
 public:
  virtual ~LQSolver() {}

  // Solve underlying LQ game to a Nash equilibrium. This will differ in derived
  // classes depending on the information structure of the game.
  // Optionally return delta xs and costates.
  virtual std::vector<Strategy> Solve(
      const std::vector<LinearDynamicsApproximation>& linearization,
      const std::vector<std::vector<QuadraticCostApproximation>>&
          quadraticization,
      const VectorXf& x0, std::vector<VectorXf>* delta_xs = nullptr,
      std::vector<std::vector<VectorXf>>* costates = nullptr) = 0;

 protected:
  LQSolver(const std::shared_ptr<const MultiPlayerIntegrableSystem>& dynamics,
           size_t num_time_steps)
      : dynamics_(dynamics), num_time_steps_(num_time_steps) {
    CHECK_NOTNULL(dynamics.get());
  }

  // Dynamics and number of time steps.
  const std::shared_ptr<const MultiPlayerIntegrableSystem> dynamics_;
  const size_t num_time_steps_;
};  // class LQSolver

}  // namespace ilqgames

#endif
