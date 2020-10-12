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
// Utility for solving a problem using a receding horizon, simulating dynamics
// forward at each stage to account for the passage of time and also switching
// to a minimally-invasive control *for only the ego vehicle* if its safety
// problem detects iminent danger.
//
// This class is intended as a facsimile of a real-time, online receding horizon
// problem in which short horizon problems are solved asynchronously throughout
// operation.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/examples/minimally_invasive_receding_horizon_simulator.h>
#include <ilqgames/solver/ilq_solver.h>
#include <ilqgames/solver/problem.h>
#include <ilqgames/solver/solution_splicer.h>
#include <ilqgames/utils/solver_log.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <chrono>
#include <memory>
#include <typeinfo>
#include <vector>

namespace ilqgames {

using clock = std::chrono::system_clock;

std::vector<ActiveProblem> MinimallyInvasiveRecedingHorizonSimulator(
    Time final_time, Time planner_runtime, GameSolver* original,
    GameSolver* safety,
    std::vector<std::shared_ptr<const SolverLog>>* original_logs,
    std::vector<std::shared_ptr<const SolverLog>>* safety_logs) {
  CHECK_NOTNULL(original);
  CHECK_NOTNULL(safety);
  CHECK_NOTNULL(original_logs);
  CHECK_NOTNULL(safety_logs);

  // Make sure the two problems have the same initial condition and time.
  CHECK(original->GetProblem().InitialState().isApprox(
      safety->GetProblem().InitialState(), constants::kSmallNumber));
  CHECK_NEAR(original->GetProblem().InitialTime(),
             safety->GetProblem().InitialTime(), constants::kSmallNumber);

  // Unpack dynamics, and ensure that the two problems actually share the same
  // dynamics object type.
  const auto& dynamics = *original->GetProblem().Dynamics();
  const auto& safety_dynamics = *safety->GetProblem().Dynamics();
  CHECK(typeid(dynamics) == typeid(safety_dynamics));

  // Clear out the log arrays for us to save in.
  original_logs->clear();
  safety_logs->clear();

  // Initial run of the solver. Keep track of time in order to know how much to
  // integrate dynamics forward. Ensure that both solvers succeed at the first
  // invocation.
  auto solver_call_time = clock::now();
  bool success = false;
  original_logs->push_back(original->Solve(&success));
  CHECK(success);
  Time elapsed_time =
      std::chrono::duration<Time>(clock::now() - solver_call_time).count();
  VLOG(1) << "Solved initial original problem in " << elapsed_time
          << " seconds, with " << original_logs->back()->NumIterates()
          << " iterations.";

  solver_call_time = clock::now();
  safety_logs->push_back(safety->Solve(&success));
  CHECK(success);
  elapsed_time =
      std::chrono::duration<Time>(clock::now() - solver_call_time).count();
  VLOG(1) << "Solved initial safety problem in " << elapsed_time
          << " seconds, with " << safety_logs->back()->NumIterates()
          << " iterations.";

  // Keep a solution splicer to incorporate new receding horizon solutions.
  // NOTE: by default, this always just starts with the original controller.
  SolutionSplicer splicer(*original_logs->front());
  std::vector<ActiveProblem> active_problem = {ActiveProblem::ORIGINAL};

  // Repeatedly integrate dynamics forward, reset original_problem initial
  // conditions, and resolve.
  VectorXf x(original->GetProblem().InitialState());
  Time t = original->GetProblem().InitialTime();

  while (true) {
    // Break the loop if it's been long enough.
    // Integrate a little more.
    constexpr Time kExtraTime = 0.25;
    t += kExtraTime;  // + planner_runtime;

    if (t >= final_time ||
        !splicer.ContainsTime(t + planner_runtime +
                              time::kTimeStep))
      break;

    x = dynamics.Integrate(t - kExtraTime, t, x,
                           splicer.CurrentOperatingPoint(),
                           splicer.CurrentStrategies());

    // Find the active problem.
    const bool current_active_problem_flag = active_problem.back();
    auto current_active_problem =
        (current_active_problem_flag == ActiveProblem::ORIGINAL) ? original
                                                                 : safety;

    // Make sure both problems have the current solution from the splicer.
    original->GetProblem().OverwriteSolution(splicer.CurrentOperatingPoint(),
                                             splicer.CurrentStrategies());
    safety->GetProblem().OverwriteSolution(splicer.CurrentOperatingPoint(),
                                           splicer.CurrentStrategies());

    // Make sure both problems have the active problem's initial state.
    original->GetProblem().ResetInitialState(
        current_active_problem->GetProblem().InitialState());
    safety->GetProblem().ResetInitialState(
        current_active_problem->GetProblem().InitialState());

    // Set up next receding horizon problem and solve, and make sure both
    // problems' initial state matches that of the active problem.
    original->GetProblem().SetUpNextRecedingHorizon(x, t, planner_runtime);
    safety->GetProblem().SetUpNextRecedingHorizon(x, t, planner_runtime);

    solver_call_time = clock::now();
    original_logs->push_back(original->Solve(&success, planner_runtime));
    const Time original_elapsed_time =
        std::chrono::duration<Time>(clock::now() - solver_call_time).count();

    CHECK_LE(original_elapsed_time, planner_runtime);
    VLOG(1) << "t = " << t << ": Solved warm-started original problem in "
            << original_elapsed_time << " seconds.";

    solver_call_time = clock::now();
    safety_logs->push_back(safety->Solve(&success, planner_runtime));
    const Time safety_elapsed_time =
        std::chrono::duration<Time>(clock::now() - solver_call_time).count();

    CHECK_LE(safety_elapsed_time, planner_runtime);
    VLOG(1) << "t = " << t << ": Solved warm-started safety problem in "
            << safety_elapsed_time << " seconds.";

    // Break the loop if it's been long enough.
    elapsed_time = std::max(original_elapsed_time, safety_elapsed_time);
    t += elapsed_time;
    if (t >= final_time || !splicer.ContainsTime(t)) break;

    // Integrate dynamics forward to account for solve time.
    x = dynamics.Integrate(t - elapsed_time, t, x,
                           splicer.CurrentOperatingPoint(),
                           splicer.CurrentStrategies());

    // Check if problems converged.
    if (!original_logs->back()->WasConverged())
      VLOG(2) << "Original planner was not converged.";
    if (!safety_logs->back()->WasConverged())
      VLOG(2) << "Safety planner was not converged.";

    // Check the safety criterion, i.e., if the safety problem's value function
    // for P1 is above kSafetyThreshold (which usually has the units of meters).
    // That said, make sure the next planner has converged if at all possible.
    constexpr float kSafetyThreshold = -1.0;
    const float p1_safety_cost = safety_logs->back()->TotalCosts().front();

    if (p1_safety_cost > kSafetyThreshold ||
        (safety_logs->back()->WasConverged() &&
         !original_logs->back()->WasConverged())) {
      active_problem.push_back(ActiveProblem::SAFETY);
      splicer.Splice(*safety_logs->back());
      VLOG(2) << "Using safety controller.";
    } else {
      active_problem.push_back(ActiveProblem::ORIGINAL);
      if (original_logs->back()->WasConverged())
        splicer.Splice(*original_logs->back());
    }
  }

  return active_problem;
}

}  // namespace ilqgames
