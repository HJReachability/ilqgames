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
// equilibrium.
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

bool NumericalCheckLocalNashEquilibrium(
    const std::vector<PlayerCost>& player_costs,
    const std::vector<Strategy>& strategies,
    const OperatingPoint& operating_point,
    const MultiPlayerIntegrableSystem& dynamics, const VectorXf& x0,
    float max_perturbation, bool open_loop) {
  CHECK_EQ(strategies.size(), player_costs.size());
  CHECK_EQ(strategies.size(), dynamics.NumPlayers());
  CHECK_EQ(x0.size(), dynamics.XDim());

  const size_t num_time_steps = strategies[0].Ps.size();
  CHECK_EQ(num_time_steps, strategies[0].alphas.size());

  // Compute nominal equilibrium cost and be sure to use only 1-step Euler
  // integration.
  const bool was_integrating_using_euler =
      MultiPlayerIntegrableSystem::IntegrationUsesEuler();
  if (!was_integrating_using_euler)
    MultiPlayerIntegrableSystem::IntegrateUsingEuler();
  const std::vector<float> nominal_costs = ComputeStrategyCosts(
      player_costs, strategies, operating_point, dynamics, x0, open_loop);

  // For each player, perturb strategies with Gaussian noise a bunch of times
  // and if cost decreases then return false.
  std::vector<Strategy> perturbed_strategies_lower(strategies);
  std::vector<Strategy> perturbed_strategies_upper(strategies);
  for (PlayerIndex ii = 0; ii < dynamics.NumPlayers(); ii++) {
    for (size_t kk = 0; kk < num_time_steps - 1; kk++) {
      VectorXf& alphak_lower = perturbed_strategies_lower[ii].alphas[kk];
      VectorXf& alphak_upper = perturbed_strategies_upper[ii].alphas[kk];

      for (size_t jj = 0; jj < alphak_lower.size(); jj++) {
        alphak_lower(jj) -= max_perturbation;
        alphak_upper(jj) += max_perturbation;

        // Compute new costs.
        const std::vector<float> perturbed_costs_lower =
            ComputeStrategyCosts(player_costs, perturbed_strategies_lower,
                                 operating_point, dynamics, x0, open_loop);
        const std::vector<float> perturbed_costs_upper =
            ComputeStrategyCosts(player_costs, perturbed_strategies_upper,
                                 operating_point, dynamics, x0, open_loop);

        // Check Nash condition.
        if (std::min(perturbed_costs_lower[ii], perturbed_costs_upper[ii]) <
            nominal_costs[ii]) {
          // std::printf(
          //     "player %hu, timestep %zu: nominal %f > perturbed %f\n ", ii,
          //     kk, nominal_costs[ii], std::min(perturbed_costs_lower[ii],
          //     perturbed_costs_upper[ii]));
          // std::cout << "nominal u: " <<
          // operating_point.us[kk][ii].transpose()
          //           << ", alpha original: "
          //           << strategies[ii].alphas[kk].transpose()
          //           << ", vs. perturbed " << alphak_lower.transpose()
          //           << std::endl;

          // Other users will likely want RK4 integration.
          if (!was_integrating_using_euler)
            MultiPlayerIntegrableSystem::IntegrateUsingRK4();
          return false;
        }

        // Reset this alpha.
        alphak_lower = strategies[ii].alphas[kk];
        alphak_upper = strategies[ii].alphas[kk];
      }
    }
  }

  // Other users will likely want RK4 integration.
  MultiPlayerIntegrableSystem::IntegrateUsingRK4();
  return true;
}

bool NumericalCheckLocalNashEquilibrium(const Problem& problem,
                                        float max_perturbation,
                                        bool open_loop) {
  return NumericalCheckLocalNashEquilibrium(
      problem.PlayerCosts(), problem.CurrentStrategies(),
      problem.CurrentOperatingPoint(), *problem.Dynamics(),
      problem.InitialState(), max_perturbation, open_loop);
}

bool CheckSufficientLocalNashEquilibrium(
    const std::vector<PlayerCost>& player_costs,
    const OperatingPoint& operating_point,
    const std::shared_ptr<const MultiPlayerIntegrableSystem> dynamics) {
  // Unpack number of players and number of time steps.
  const PlayerIndex num_players = player_costs.size();
  const size_t num_time_steps = operating_point.xs.size();
  const Dimension xdim = operating_point.xs[0].size();

  // Set up quadratic cost approximations.
  std::vector<QuadraticCostApproximation> quadraticization(
      num_players, QuadraticCostApproximation(xdim));

  // Quadraticize costs and check PSD conditions.
  for (size_t kk = 0; kk < num_time_steps; kk++) {
    const Time t = operating_point.t0 + static_cast<Time>(kk) * time::kTimeStep;
    VectorXf x = operating_point.xs[kk];
    std::vector<VectorXf> us = operating_point.us[kk];

    // Maybe convert out of linear system coordinates.
    if (dynamics.get() && dynamics->TreatAsLinear()) {
      const auto& dyn =
          *static_cast<const MultiPlayerFlatSystem*>(dynamics.get());

      // Previous x, us are actually xi, vs.
      x = dyn.FromLinearSystemState(x.eval());
      us = dyn.LinearizingControls(x, std::vector<VectorXf>(us));
    }

    std::transform(player_costs.begin(), player_costs.end(),
                   quadraticization.begin(),
                   [&t, &x, &us](const PlayerCost& cost) {
                     return cost.Quadraticize(t, x, us);
                   });

    // Check if Q, Rs PSD.
    constexpr float kErrorMargin = 1e-4;
    for (const auto& q : quadraticization) {
      const auto eig_Q = Eigen::SelfAdjointEigenSolver<MatrixXf>(q.state.hess);
      if (eig_Q.eigenvalues().minCoeff() < -kErrorMargin) {
        // std::cout << "Failed at timestep " << kk << std::endl;
        // std::cout << "Q is: \n" << q.Q << std::endl;
        // std::cout << "Q evals are: " << eig_Q.eigenvalues().transpose()
        //           << std::endl;
        return false;
      }

      for (const auto& entry : q.control) {
        const auto eig_R =
            Eigen::SelfAdjointEigenSolver<MatrixXf>(entry.second.hess);
        if (eig_R.eigenvalues().minCoeff() < -kErrorMargin) return false;
      }
    }
  }

  return true;
}

bool CheckSufficientLocalNashEquilibrium(const Problem& problem) {
  return CheckSufficientLocalNashEquilibrium(problem.PlayerCosts(),
                                             problem.CurrentOperatingPoint(),
                                             problem.Dynamics());
}

}  // namespace ilqgames
