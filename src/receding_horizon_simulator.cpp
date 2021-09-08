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
// Utility for solving a problem using a receding horizon, simulating dynamics
// forward at each stage to account for the passage of time.
//
// This class is intended as a facsimile of a real-time, online receding horizon
// problem in which short horizon problems are solved asynchronously throughout
// operation.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/examples/receding_horizon_simulator.h>
#include <ilqgames/solver/ilq_solver.h>
#include <ilqgames/solver/problem.h>
#include <ilqgames/solver/solution_splicer.h>
#include <ilqgames/utils/solver_log.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <chrono>
#include <memory>
#include <vector>

namespace ilqgames {

using clock = std::chrono::system_clock;

std::vector<std::shared_ptr<const SolverLog>> RecedingHorizonSimulator(
    Time final_time, Time planner_runtime, GameSolver* solver) {
  CHECK_NOTNULL(solver);

  // Set up a list of solver logs, one per solver invocation.
  std::vector<std::shared_ptr<const SolverLog>> logs;

  // Initial run of the solver. Keep track of time in order to know how much to
  // integrate dynamics forward.
  auto solver_call_time = clock::now();
  bool success = false;
  logs.push_back(solver->Solve(&success));
  CHECK(success);
  Time elapsed_time =
      std::chrono::duration<Time>(clock::now() - solver_call_time).count();

  VLOG(1) << "Solved initial problem in " << elapsed_time << " seconds, with "
          << logs.back()->NumIterates() << " iterations.";
  const auto& dynamics = solver->GetProblem().Dynamics();

  // Keep a solution splicer to incorporate new receding horizon solutions.
  SolutionSplicer splicer(*logs.front());

  // Repeatedly integrate dynamics forward, reset problem initial conditions,
  // and resolve.
  VectorXf x(solver->GetProblem().InitialState());
  Time t = splicer.CurrentOperatingPoint().t0;

  while (true) {
    // Break the loop if it's been long enough.
    // Integrate a little more.
    constexpr Time kExtraTime = 0.25;
    t += kExtraTime;  // + planner_runtime;

    if (t >= final_time ||
        !splicer.ContainsTime(t + planner_runtime + time::kTimeStep))
      break;

    x = solver->GetProblem().Dynamics()->Integrate(
        t - kExtraTime, t, x, splicer.CurrentOperatingPoint(),
        splicer.CurrentStrategies());

    // Overwrite problem with spliced solution.
    solver->GetProblem().OverwriteSolution(splicer.CurrentOperatingPoint(),
                                           splicer.CurrentStrategies());

    // Set up next receding horizon problem and solve.
    solver->GetProblem().SetUpNextRecedingHorizon(x, t, planner_runtime);

    solver_call_time = clock::now();
    logs.push_back(solver->Solve(&success, planner_runtime));
    elapsed_time =
        std::chrono::duration<Time>(clock::now() - solver_call_time).count();

    CHECK_LE(elapsed_time, planner_runtime);
    VLOG(1) << "t = " << t << ": Solved warm-started problem in "
            << elapsed_time << " seconds.";

    // Break the loop if it's been long enough.
    t += elapsed_time;
    if (t >= final_time || !splicer.ContainsTime(t)) break;

    // Integrate dynamics forward to account for solve time.
    x = solver->GetProblem().Dynamics()->Integrate(
        t - elapsed_time, t, x, splicer.CurrentOperatingPoint(),
        splicer.CurrentStrategies());

    // Add new solution to splicer if it converged.
    if (logs.back()->WasConverged()) splicer.Splice(*logs.back());
  }

  return logs;
}

}  // namespace ilqgames
