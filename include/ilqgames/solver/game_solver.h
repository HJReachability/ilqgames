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
// Base class for all game solvers. All solvers will need linearization,
// quadraticization, and loop timing.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_SOLVER_GAME_SOLVER_H
#define ILQGAMES_SOLVER_GAME_SOLVER_H

#include <ilqgames/dynamics/multi_player_dynamical_system.h>
#include <ilqgames/dynamics/multi_player_integrable_system.h>
#include <ilqgames/solver/lq_feedback_solver.h>
#include <ilqgames/solver/lq_open_loop_solver.h>
#include <ilqgames/solver/lq_solver.h>
#include <ilqgames/solver/problem.h>
#include <ilqgames/solver/solver_params.h>
#include <ilqgames/utils/linear_dynamics_approximation.h>
#include <ilqgames/utils/loop_timer.h>
#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/quadratic_cost_approximation.h>
#include <ilqgames/utils/solver_log.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <chrono>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace ilqgames {

namespace {

// Maximum number of loop times to store in loop timer.
static constexpr size_t kMaxLoopTimesToRecord = 10;

}  // anonymous namespace

class GameSolver {
 public:
  virtual ~GameSolver() {}

  // Solve this game. Returns true if converged.
  virtual std::shared_ptr<SolverLog> Solve(
      bool* success = nullptr, Time max_runtime = constants::kInfinity) = 0;

  // Accessors.
  Problem& GetProblem() { return *problem_; }

 protected:
  GameSolver(const std::shared_ptr<Problem>& problem,
             const SolverParams& params)
      : problem_(problem), params_(params), timer_(kMaxLoopTimesToRecord) {
    CHECK_NOTNULL(problem_.get());
    CHECK_NOTNULL(problem_->Dynamics().get());
  }

  // Create a new log. This may be overridden by derived classes (e.g., to
  // change the name of the log).
  virtual std::shared_ptr<SolverLog> CreateNewLog() const {
    return std::make_shared<SolverLog>();
  }

  // Store the underlying problem.
  const std::shared_ptr<Problem> problem_;

  // Solver parameters.
  const SolverParams params_;

  // Timer to keep track of loop execution times.
  LoopTimer timer_;
};  // class GameSolver

}  // namespace ilqgames

#endif
