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
// Check whether or not a particular set of strategies is a local Nash
// equilibrium. Since we do not have easy access to gradients and Hessians of
// each players' total cost with respect to Ps and alphas (though we do have
// such information at each time step), we shall resort to random sampling.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/cost/player_cost.h>
#include <ilqgames/dynamics/multi_player_dynamical_system.h>
#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <random>
#include <vector>

namespace ilqgames {

namespace {
// Compute cost of a set of strategies for each player.
std::vector<float> ComputeStrategyCosts(
    const std::vector<PlayerCost>& player_costs,
    const std::vector<Strategy>& strategies,
    const OperatingPoint& operating_point,
    const MultiPlayerDynamicalSystem& dynamics, const VectorXf& x0,
    float time_step) {
  const size_t num_time_steps = strategies[0].Ps.size();
  CHECK_EQ(num_time_steps, strategies[0].alphas.size());

  // Start at the initial state.
  VectorXf x(x0);
  Time t = 0.0;

  // Walk forward along the trajectory and accumulate total cost.
  std::vector<VectorXf> us(dynamics.NumPlayers());
  std::vector<float> total_costs(dynamics.NumPlayers(), 0.0);
  for (size_t kk = 0; kk < num_time_steps; kk++) {
    // Update controls.
    for (PlayerIndex ii = 0; ii < dynamics.NumPlayers(); ii++) {
      us[ii] = strategies[ii](kk, x - operating_point.xs[kk],
                              operating_point.us[kk][ii]);
    }

    // Update costs.
    for (PlayerIndex ii = 0; ii < dynamics.NumPlayers(); ii++)
      total_costs[ii] += player_costs[ii].Evaluate(t, x, us);

    // Update state and time
    x = dynamics.Integrate(t, time_step, x, us);
    t += time_step;
  }

  return total_costs;
}

}  // anonymous namespace

bool CheckLocalNashEquilibrium(const std::vector<PlayerCost>& player_costs,
                               const std::vector<Strategy>& strategies,
                               const OperatingPoint& operating_point,
                               const MultiPlayerDynamicalSystem& dynamics,
                               const VectorXf& x0, float time_step,
                               float gaussian_perturbation_stddev,
                               size_t num_perturbations_per_player) {
  CHECK_EQ(strategies.size(), player_costs.size());
  CHECK_EQ(strategies.size(), dynamics.NumPlayers());
  CHECK_EQ(x0.size(), dynamics.XDim());

  // Set up random number generator.
  std::random_device rd;
  std::default_random_engine rng(rd());
  std::normal_distribution<float> gaussian(0.0, gaussian_perturbation_stddev);

  // Compute nominal equilibrium cost.
  const std::vector<float> nominal_costs = ComputeStrategyCosts(
      player_costs, strategies, operating_point, dynamics, x0, time_step);

  // For each player, perturb strategies with Gaussian noise a bunch of times
  // and if cost decreases then return false.
  // TODO!

  return true;
}

}  // namespace ilqgames
