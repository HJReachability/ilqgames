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

#include <ilqgames/examples/defensive_driving_receding_horizon_simulator.h>
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

void DefensiveDrivingRecedingHorizonSimulator(
    Time final_time, Time planner_runtime, GameSolver* defensive,
    GameSolver* normal,
    std::vector<std::shared_ptr<const SolverLog>>* defensive_logs,
    std::vector<std::shared_ptr<const SolverLog>>* normal_logs) {
  CHECK_NOTNULL(defensive);
  CHECK_NOTNULL(normal);
  CHECK_NOTNULL(defensive_logs);
  CHECK_NOTNULL(normal_logs);

  // Make sure the two problems have the same initial condition and time.
  CHECK(defensive->GetProblem().InitialState().isApprox(
      normal->GetProblem().InitialState(), constants::kSmallNumber));
  CHECK_NEAR(defensive->GetProblem().InitialTime(),
             normal->GetProblem().InitialTime(), constants::kSmallNumber);

  // Unpack dynamics, and ensure that the two problems actually share the same
  // dynamics object type.
  const auto& dynamics = *defensive->GetProblem().Dynamics();
  const auto& normal_dynamics = *normal->GetProblem().Dynamics();
  CHECK(typeid(dynamics) == typeid(normal_dynamics));

  // Clear out the log arrays for us to save in.
  defensive_logs->clear();
  normal_logs->clear();

  // Initial run of the solver. Ensure that both solvers succeed at the first
  // invocation.
  auto solver_call_time = Clock::now();
  bool success = false;
  defensive_logs->push_back(defensive->Solve(&success));
  CHECK(success);
  Time elapsed_time =
      std::chrono::duration<Time>(Clock::now() - solver_call_time).count();
  VLOG(1) << "Solved initial defensive problem in " << elapsed_time
          << " seconds, with " << defensive_logs->back()->NumIterates()
          << " iterations.";

  solver_call_time = Clock::now();
  normal_logs->push_back(normal->Solve(&success));
  CHECK(success);
  elapsed_time =
      std::chrono::duration<Time>(Clock::now() - solver_call_time).count();
  VLOG(1) << "Solved initial normal problem in " << elapsed_time
          << " seconds, with " << normal_logs->back()->NumIterates()
          << " iterations.";

  // Keep a solution splicer to incorporate new receding horizon solutions.
  // NOTE: by default, this always just starts with the defensive controller.
  const OperatingPoint stitched_op =
      dynamics.Stitch(defensive_logs->back()->FinalOperatingPoint(),
                      normal_logs->back()->FinalOperatingPoint());
  const std::vector<Strategy> stitched_strategies =
      dynamics.Stitch(defensive_logs->back()->FinalStrategies(),
                      normal_logs->back()->FinalStrategies());
  SolutionSplicer splicer(stitched_op, stitched_strategies);

  // Repeatedly integrate dynamics forward, reset defensive_problem initial
  // conditions, and resolve.
  VectorXf x(defensive->GetProblem().InitialState());
  Time t = defensive->GetProblem().InitialTime();

  while (true) {
    // Break the loop if it's been long enough.
    // Integrate a little more.
    constexpr Time kExtraTime = 0.25;
    t += kExtraTime;  // + planner_runtime;

    if (t >= final_time ||
        !splicer.ContainsTime(t + planner_runtime + time::kTimeStep))
      break;

    x = dynamics.Integrate(t - kExtraTime, t, x,
                           splicer.CurrentOperatingPoint(),
                           splicer.CurrentStrategies());

    // Make sure both problems have the current solution from the splicer.
    defensive->GetProblem().OverwriteSolution(splicer.CurrentOperatingPoint(),
                                              splicer.CurrentStrategies());
    normal->GetProblem().OverwriteSolution(splicer.CurrentOperatingPoint(),
                                           splicer.CurrentStrategies());

    // Set up next receding horizon problem and solve, and make sure both
    // problems' initial state matches that of the active problem.
    defensive->GetProblem().SetUpNextRecedingHorizon(x, t, planner_runtime);
    normal->GetProblem().SetUpNextRecedingHorizon(x, t, planner_runtime);

    solver_call_time = Clock::now();
    defensive_logs->push_back(defensive->Solve(&success, planner_runtime));
    const Time defensive_elapsed_time =
        std::chrono::duration<Time>(Clock::now() - solver_call_time).count();

    CHECK_LE(defensive_elapsed_time, planner_runtime);
    VLOG(1) << "t = " << t << ": Solved warm-started defensive problem in "
            << defensive_elapsed_time << " seconds.";

    solver_call_time = Clock::now();
    normal_logs->push_back(normal->Solve(&success, planner_runtime));
    const Time normal_elapsed_time =
        std::chrono::duration<Time>(Clock::now() - solver_call_time).count();

    CHECK_LE(normal_elapsed_time, planner_runtime);
    VLOG(1) << "t = " << t << ": Solved warm-started normal problem in "
            << normal_elapsed_time << " seconds.";

    // Break the loop if it's been long enough.
    elapsed_time = std::max(defensive_elapsed_time, normal_elapsed_time);
    t += elapsed_time;
    if (t >= final_time || !splicer.ContainsTime(t)) break;

    // Integrate dynamics forward to account for solve time.
    x = dynamics.Integrate(t - elapsed_time, t, x,
                           splicer.CurrentOperatingPoint(),
                           splicer.CurrentStrategies());

    // Check if problems converged.
    if (!defensive_logs->back()->WasConverged())
      VLOG(2) << "Defensive planner was not converged.";
    if (!normal_logs->back()->WasConverged())
      VLOG(2) << "Normal planner was not converged.";

    // Splice in new solutions.
    const OperatingPoint stitched_op =
        dynamics.Stitch(defensive_logs->back()->FinalOperatingPoint(),
                        normal_logs->back()->FinalOperatingPoint());
    const std::vector<Strategy> stitched_strategies =
        dynamics.Stitch(defensive_logs->back()->FinalStrategies(),
                        normal_logs->back()->FinalStrategies());
    splicer.Splice(stitched_op, stitched_strategies);
  }
}

}  // namespace ilqgames
