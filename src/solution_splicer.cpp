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
// Splice together existing and new solutions to a receding horizon problem.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/dynamics/multi_player_integrable_system.h>
#include <ilqgames/solver/solution_splicer.h>
#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/solver_log.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <memory>
#include <vector>

namespace ilqgames {

SolutionSplicer::SolutionSplicer(const SolverLog& log)
    : strategies_(log.FinalStrategies()),
      operating_point_(log.FinalOperatingPoint()) {}

void SolutionSplicer::Splice(const SolverLog& log) {
  CHECK_GE(log.FinalOperatingPoint().t0, operating_point_.t0);
  CHECK_GE(operating_point_.xs.size(), time::kNumTimeSteps);
  CHECK_EQ(log.FinalOperatingPoint().xs.size(), time::kNumTimeSteps);

  const size_t current_timestep = static_cast<size_t>(
      1e-4 +  // Add a little so that conversion doesn't end up subtracting 1.
      (log.FinalOperatingPoint().t0 - operating_point_.t0) / time::kTimeStep);

  // HACK! If we're close enough to the beginning of the old trajectory, just
  // save the first few steps along it in case a lower-level path follower uses
  // this information.
  constexpr size_t kNumPreviousTimeStepsToSave = 5;
  const size_t initial_timestep =
      (static_cast<int>(current_timestep) <
       static_cast<int>(kNumPreviousTimeStepsToSave))
          ? 0
          : current_timestep - kNumPreviousTimeStepsToSave;

  // HACK! Make sure the new solution starts several timesteps after the
  // nearest match to guard against off-by-one issues.
  constexpr size_t kNumExtraTimeStepsBeforeSplicingIn = 0;
  const size_t first_timestep_new_solution =
      kNumExtraTimeStepsBeforeSplicingIn + current_timestep;

  // (2) Copy over saved part of existing plan.
  for (size_t kk = initial_timestep; kk < first_timestep_new_solution; kk++) {
    const size_t kk_new_solution = kk - initial_timestep;
    operating_point_.xs[kk_new_solution].swap(operating_point_.xs[kk]);
    operating_point_.us[kk_new_solution].swap(operating_point_.us[kk]);

    for (auto& strategy : strategies_) {
      strategy.Ps[kk_new_solution].swap(strategy.Ps[kk]);
      strategy.alphas[kk_new_solution].swap(strategy.alphas[kk]);
    }
  }

  // Resize to be the appropriate length.
  // NOTE: makes use of default behavior of std::vector<T>.resize() in that it
  // does not delete earlier entries.
  const size_t num_spliced_timesteps =
      current_timestep - initial_timestep + time::kNumTimeSteps;
  CHECK_LE(num_spliced_timesteps,
           time::kNumTimeSteps + kNumPreviousTimeStepsToSave);

  operating_point_.xs.resize(num_spliced_timesteps);
  operating_point_.us.resize(num_spliced_timesteps);
  operating_point_.t0 += initial_timestep * time::kTimeStep;

  for (auto& strategy : strategies_) {
    strategy.Ps.resize(num_spliced_timesteps);
    strategy.alphas.resize(num_spliced_timesteps);
  }

  // Copy over new solution to overwrite existing log after first timestep.
  CHECK_EQ(current_timestep + time::kNumTimeSteps - initial_timestep,
           operating_point_.xs.size());
  for (size_t kk = kNumExtraTimeStepsBeforeSplicingIn; kk < time::kNumTimeSteps;
       kk++) {
    const size_t kk_new_solution = current_timestep + kk - initial_timestep;
    operating_point_.xs[kk_new_solution] = log.FinalOperatingPoint().xs[kk];
    operating_point_.us[kk_new_solution] = log.FinalOperatingPoint().us[kk];

    for (PlayerIndex ii = 0; ii < log.NumPlayers(); ii++) {
      strategies_[ii].Ps[kk_new_solution] = log.FinalStrategies()[ii].Ps[kk];
      strategies_[ii].alphas[kk_new_solution] =
          log.FinalStrategies()[ii].alphas[kk];
    }
  }
}

}  // namespace ilqgames
