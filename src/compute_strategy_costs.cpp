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
// Compute costs for each player associated with this set of strategies and
// operating point.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/cost/player_cost.h>
#include <ilqgames/dynamics/multi_player_flat_system.h>
#include <ilqgames/dynamics/multi_player_integrable_system.h>
#include <ilqgames/utils/compute_strategy_costs.h>
#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/quadratic_cost_approximation.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <Eigen/Dense>
#include <random>
#include <vector>

namespace ilqgames {

// Compute cost of a set of strategies for each player.
std::vector<float> ComputeStrategyCosts(
    const std::vector<PlayerCost>& player_costs,
    const std::vector<Strategy>& strategies,
    const OperatingPoint& operating_point,
    const MultiPlayerIntegrableSystem& dynamics, const VectorXf& x0,
    bool open_loop) {
  // Start at the initial state.
  VectorXf x(x0);
  Time t = 0.0;

  // Walk forward along the trajectory and accumulate total cost.
  std::vector<VectorXf> us(dynamics.NumPlayers());
  std::vector<float> total_costs(dynamics.NumPlayers(), 0.0);
  const size_t num_time_steps =
      (open_loop) ? strategies[0].Ps.size() - 1 : strategies[0].Ps.size();
  for (size_t kk = 0; kk < num_time_steps; kk++) {
    // Update controls.
    for (PlayerIndex ii = 0; ii < dynamics.NumPlayers(); ii++) {
      if (open_loop)
        us[ii] = strategies[ii](kk, VectorXf::Zero(x.size()),
                                operating_point.us[kk][ii]);
      else
        us[ii] = strategies[ii](kk, x - operating_point.xs[kk],
                                operating_point.us[kk][ii]);
    }

    const VectorXf next_x = dynamics.Integrate(t, time::kTimeStep, x, us);
    const Time next_t = t + time::kTimeStep;

    // Update costs.
    for (PlayerIndex ii = 0; ii < dynamics.NumPlayers(); ii++) {
      const float cost =
          (open_loop) ? player_costs[ii].EvaluateOffset(t, next_t, next_x, us)
                      : player_costs[ii].Evaluate(t, x, us);

      total_costs[ii] += cost;
    }

    // Update state and time
    x = next_x;
    t = next_t;
  }

  return total_costs;
}

std::vector<float> ComputeStrategyCosts(const Problem& problem,
                                        bool open_loop) {
  return ComputeStrategyCosts(
      problem.PlayerCosts(), problem.CurrentStrategies(),
      problem.CurrentOperatingPoint(), *problem.Dynamics(),
      problem.InitialState(), open_loop);
}

}  // namespace ilqgames
