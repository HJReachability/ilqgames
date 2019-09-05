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

#include <ilqgames/dynamics/multi_player_dynamical_system.h>
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

void SolutionSplicer::Splice(const SolverLog& log, Time current_time) {
  CHECK_LE(current_time, log.InitialTime());
  CHECK_GE(current_time, operating_point_.t0);

  // (1) Identify current timestep and first timestep of new solution.
  const size_t current_timestep = static_cast<size_t>(
      (current_time - operating_point_.t0) / log.TimeStep());
  const size_t first_timestep_new_solution = static_cast<size_t>(
      (log.InitialTime() - operating_point_.t0) / log.TimeStep());

  // Resize to be the appropriate length.
  const size_t num_spliced_timesteps =
      first_timestep_new_solution - current_timestep + log.NumTimeSteps();
  operating_point_.xs.resize(num_spliced_timesteps);
  operating_point_.us.resize(num_spliced_timesteps);
  operating_point_.t0 += current_timestep * log.TimeStep();

  for (auto& strategy : strategies_) {
    strategy.Ps.resize(num_spliced_timesteps);
    strategy.alphas.resize(num_spliced_timesteps);
  }

  // (2) Prune old part of existing plan.
  for (size_t kk = current_timestep; kk < first_timestep_new_solution; kk++) {
    const size_t kk_new_solution = kk - current_timestep;
    operating_point_.xs[kk_new_solution] = operating_point_.xs[kk];
    operating_point_.us[kk_new_solution] = operating_point_.us[kk];

    for (auto& strategy : strategies_) {
      strategy.Ps[kk_new_solution] = strategy.Ps[kk];
      strategy.alphas[kk_new_solution] = strategy.alphas[kk];
    }
  }

  // (3) Copy over new solution to overwrite existing log after first timestep.
  CHECK_EQ(first_timestep_new_solution - current_timestep + log.NumTimeSteps(),
           operating_point_.xs.size());
  for (size_t kk = 0; kk < log.NumTimeSteps(); kk++) {
    const size_t kk_new_solution =
        kk + first_timestep_new_solution - current_timestep;
    DCHECK_LT(kk_new_solution, operating_point_.xs.size());
    DCHECK_LT(kk_new_solution, operating_point_.us.size());
    DCHECK_LT(kk, log.FinalOperatingPoint().xs.size());
    DCHECK_LT(kk, log.FinalOperatingPoint().us.size());
    operating_point_.xs[kk_new_solution] = log.FinalOperatingPoint().xs[kk];
    operating_point_.us[kk_new_solution] = log.FinalOperatingPoint().us[kk];

    DCHECK_EQ(log.NumPlayers(), strategies_.size());
    DCHECK_EQ(log.NumPlayers(), log.FinalStrategies().size());
    for (PlayerIndex ii = 0; ii < log.NumPlayers(); ii++) {
      DCHECK_LT(kk_new_solution, strategies_[ii].Ps.size());
      DCHECK_LT(kk_new_solution, strategies_[ii].alphas.size());
      DCHECK_LT(kk, log.FinalStrategies()[ii].Ps.size());
      DCHECK_LT(kk, log.FinalStrategies()[ii].alphas.size());

      strategies_[ii].Ps[kk_new_solution] = log.FinalStrategies()[ii].Ps[kk];
      strategies_[ii].alphas[kk_new_solution] =
          log.FinalStrategies()[ii].alphas[kk];
    }
  }
}

void SolutionSplicer::Splice(const SolverLog& log, const VectorXf& x,
                             const MultiPlayerDynamicalSystem& dynamics) {
  // (1) Identify current timestep and first timestep of new solution.
  const VectorXf& new_x0 = log.FinalOperatingPoint().xs[0];
  const auto nearest_iter_new_x0 = std::min_element(
      operating_point_.xs.begin(), operating_point_.xs.end(),
      [&dynamics, &new_x0](const VectorXf& x1, const VectorXf& x2) {
        return dynamics.DistanceBetween(new_x0, x1) <
               dynamics.DistanceBetween(new_x0, x2);
      });
  const auto nearest_iter_x =
      std::min_element(operating_point_.xs.begin(), operating_point_.xs.end(),
                       [&dynamics, &x](const VectorXf& x1, const VectorXf& x2) {
                         return dynamics.DistanceBetween(x, x1) <
                                dynamics.DistanceBetween(x, x2);
                       });

  const size_t current_timestep =
      (nearest_iter_x == operating_point_.xs.begin())
          ? 0
          : std::distance(operating_point_.xs.begin(), nearest_iter_x) - 1;
  const size_t first_timestep_new_solution = std::max<size_t>(
      current_timestep,
      std::distance(operating_point_.xs.begin(), nearest_iter_new_x0));

  // Resize to be the appropriate length.
  const size_t num_spliced_timesteps =
      first_timestep_new_solution - current_timestep + log.NumTimeSteps();
  operating_point_.xs.resize(num_spliced_timesteps);
  operating_point_.us.resize(num_spliced_timesteps);
  operating_point_.t0 = log.FinalOperatingPoint().t0 -
                        first_timestep_new_solution * log.TimeStep();

  for (auto& strategy : strategies_) {
    strategy.Ps.resize(num_spliced_timesteps);
    strategy.alphas.resize(num_spliced_timesteps);
  }

  // (2) Prune old part of existing plan.
  for (size_t kk = current_timestep; kk < first_timestep_new_solution; kk++) {
    const size_t kk_new_solution = kk - current_timestep;
    operating_point_.xs[kk_new_solution] = operating_point_.xs[kk];
    operating_point_.us[kk_new_solution] = operating_point_.us[kk];

    for (auto& strategy : strategies_) {
      strategy.Ps[kk_new_solution] = strategy.Ps[kk];
      strategy.alphas[kk_new_solution] = strategy.alphas[kk];
    }
  }

  // (3) Copy over new solution to overwrite existing log after first
  // timestep.
  CHECK_EQ(first_timestep_new_solution - current_timestep + log.NumTimeSteps(),
           operating_point_.xs.size());
  for (size_t kk = 0; kk < log.NumTimeSteps(); kk++) {
    const size_t kk_new_solution =
        kk + first_timestep_new_solution - current_timestep;
    DCHECK_LT(kk_new_solution, operating_point_.xs.size());
    DCHECK_LT(kk_new_solution, operating_point_.us.size());
    DCHECK_LT(kk, log.FinalOperatingPoint().xs.size());
    DCHECK_LT(kk, log.FinalOperatingPoint().us.size());
    operating_point_.xs[kk_new_solution] = log.FinalOperatingPoint().xs[kk];
    operating_point_.us[kk_new_solution] = log.FinalOperatingPoint().us[kk];

    DCHECK_EQ(log.NumPlayers(), strategies_.size());
    DCHECK_EQ(log.NumPlayers(), log.FinalStrategies().size());
    for (PlayerIndex ii = 0; ii < log.NumPlayers(); ii++) {
      DCHECK_LT(kk_new_solution, strategies_[ii].Ps.size());
      DCHECK_LT(kk_new_solution, strategies_[ii].alphas.size());
      DCHECK_LT(kk, log.FinalStrategies()[ii].Ps.size());
      DCHECK_LT(kk, log.FinalStrategies()[ii].alphas.size());

      strategies_[ii].Ps[kk_new_solution] = log.FinalStrategies()[ii].Ps[kk];
      strategies_[ii].alphas[kk_new_solution] =
          log.FinalStrategies()[ii].alphas[kk];
    }
  }
}

}  // namespace ilqgames
