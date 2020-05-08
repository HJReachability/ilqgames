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
// Core open-loop LQ game solver based on Basar and Olsder, Chapter 6. All
// notation matches the text, though we shall assume that `c` (additive drift in
// dynamics) is always `0`, which holds because these dynamics are for delta x,
// delta us. Also, we have modified terms slightly to account for linear terms
// in the stage cost for control, i.e.
//       control penalty i = 0.5 \sum_j du_j^T R_ij (du_j + 2 r_ij)
//
// Solve a time-varying, finite horizon LQ game (finds open-loop Nash
// feedback strategies for both players).
//
// Assumes that dynamics are given by
//           ``` dx_{k+1} = A_k dx_k + \sum_i Bs[i]_k du[i]_k ```
//
// Returns strategies Ps, alphas. Here, all the Ps are zero (by default), and
// only the alphas are nonzero.
//
// Notation is based on derivation which may be found in the PDF included in
// this repository named "open_loop_lq_derivation.pdf".
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/dynamics/multi_player_integrable_system.h>
#include <ilqgames/solver/lq_feedback_solver.h>
#include <ilqgames/solver/lq_open_loop_solver.h>
#include <ilqgames/utils/linear_dynamics_approximation.h>
#include <ilqgames/utils/quadratic_cost_approximation.h>
#include <ilqgames/utils/strategy.h>

#include <glog/logging.h>
#include <Eigen/Cholesky>
#include <vector>

namespace ilqgames {

std::vector<Strategy> LQOpenLoopSolver::Solve(
    const MultiPlayerIntegrableSystem& dynamics,
    const std::vector<LinearDynamicsApproximation>& linearization,
    const std::vector<std::vector<QuadraticCostApproximation>>&
        quadraticization,
    const VectorXf& x0) {
  // Unpack horizon.
  const size_t horizon = linearization.size();
  CHECK_EQ(quadraticization.size(), horizon);
  CHECK_GT(horizon, 0);

  // List of player-indexed strategies (each of which is a time-indexed
  // affine state error-feedback controller). Since this is an open-loop
  // strategy, we will not change the default zero value of the P matrix.
  std::vector<Strategy> strategies;
  for (PlayerIndex ii = 0; ii < dynamics.NumPlayers(); ii++)
    strategies.emplace_back(horizon, dynamics.XDim(), dynamics.UDim(ii));

  // Initialize lambda^i and M^i and index first by time and then by player.
  std::vector<std::vector<VectorXf>> lambdas(horizon);
  std::vector<std::vector<MatrixXf>> Ms(horizon);
  for (PlayerIndex ii = 0; ii < dynamics.NumPlayers(); ii++) {
    lambdas.back().emplace_back(quadraticization.back()[ii].state.grad);
    Ms.back().emplace_back(quadraticization.back()[ii].state.hess);
  }

  // Instantiate the rest of the "special" terms.
  std::vector<MatrixXf> capital_lambdas(horizon - 1);
  std::vector<VectorXf> bar_lambdas(horizon - 1);
  std::vector<VectorXf> hat_lambdas(horizon - 1);

  // Precompute Cholesky decompositions and inv(R^i) * B^{iT} and inv(R^i) *
  // r^i for each player i. This will be reused, so cache the result. Do the
  // same thing for inv(Lambda) * (hat_lambda + bar_lambda) and inv(Lambda) * A.
  std::vector<std::vector<Eigen::LDLT<MatrixXf>>> chol_Rs(horizon - 1);
  std::vector<std::vector<MatrixXf>> warped_Bs(horizon - 1);
  std::vector<std::vector<VectorXf>> warped_rs(horizon - 1);
  std::vector<Eigen::LDLT<MatrixXf>> chol_capital_lambdas(horizon - 1);
  std::vector<VectorXf> warped_lambdas(horizon - 1);
  std::vector<MatrixXf> warped_As(horizon - 1);

  // (1) Work backward in time and cache "special" terms.
  // NOTE: time starts from the second-to-last entry since we'll treat the
  // final entry as a terminal cost as in Basar and Olsder, ch. 6.
  for (int kk = horizon - 2; kk >= 0; kk--) {
    // Unpack linearization and quadraticization at this time step.
    const auto& lin = linearization[kk];
    const auto& quad = quadraticization[kk];

    for (PlayerIndex ii = 0; ii < dynamics.NumPlayers(); ii++) {
      const auto control_iter = quad[ii].control.find(ii);
      CHECK(control_iter != quad[ii].control.end());

      chol_Rs[kk].emplace_back(control_iter->second.hess);
      warped_Bs[kk].emplace_back(
          chol_Rs[kk].back().solve(lin.Bs[ii].transpose()));
      warped_rs[kk].emplace_back(
          chol_Rs[kk].back().solve(control_iter->second.grad));
    }

    // Compute "special" terms.
    capital_lambdas[kk] = MatrixXf::Identity(dynamics.XDim(), dynamics.XDim());
    bar_lambdas[kk] = VectorXf::Zero(dynamics.XDim());
    hat_lambdas[kk] = VectorXf::Zero(dynamics.XDim());
    for (PlayerIndex ii = 0; ii < dynamics.NumPlayers(); ii++) {
      const MatrixXf B_warpedB_prod = lin.Bs[ii] * warped_Bs[kk][ii];
      capital_lambdas[kk] += B_warpedB_prod * Ms[kk + 1][ii];
      bar_lambdas[kk] += lin.Bs[ii] * warped_rs[kk][ii];
      hat_lambdas[kk] += B_warpedB_prod * lambdas[kk + 1][ii];
    }

    // Compute some Cholesky terms.
    chol_capital_lambdas[kk] = Eigen::LDLT<MatrixXf>(capital_lambdas[kk]);
    warped_lambdas[kk] =
        chol_capital_lambdas[kk].solve(hat_lambdas[kk] + bar_lambdas[kk]);
    warped_As[kk] = chol_capital_lambdas[kk].solve(lin.A);

    // Compute lambdas and Ms.
    for (PlayerIndex ii = 0; ii < dynamics.NumPlayers(); ii++) {
      const MatrixXf AM_product = lin.A * Ms[kk + 1][ii];
      lambdas[kk].emplace_back(
          lin.A.transpose() * quadraticization[kk + 1][ii].state.grad +
          quad[ii].state.grad - AM_product * warped_lambdas[kk]);
      Ms[kk].emplace_back(quad[ii].state.hess + AM_product * warped_As[kk]);
    }
  }

  // (2) Now compute optimal state trajectory forward in time.
  std::vector<VectorXf> xs(horizon);
  xs[0] = x0;
  for (size_t kk = 1; kk < horizon; kk++)
    xs[kk] = warped_As[kk - 1] * xs[kk - 1] - warped_lambdas[kk - 1];

  // (3) Finally, compute optimal control trajectory backward in time.
  // Set optimal control as negative alpha for each player, at each time.
  for (int kk = horizon - 2; kk >= 0; kk--) {
    for (PlayerIndex ii = 0; ii < dynamics.NumPlayers(); ii++) {
      strategies[ii].alphas[kk] =
          warped_Bs[kk][ii] *
              (Ms[kk + 1][ii] * xs[kk + 1] + lambdas[kk + 1][ii]) +
          warped_rs[kk][ii];
    }
  }

  // for (size_t kk = 1; kk < horizon; kk++) {
  //   VectorXf check_x = linearization[kk - 1].A * xs[kk - 1];
  //   for (PlayerIndex ii = 0; ii < dynamics.NumPlayers(); ii++)
  //     check_x -= linearization[kk - 1].Bs[ii] * strategies[ii].alphas[kk -
  //     1];

  //   CHECK_LE((xs[kk] - check_x).cwiseAbs().maxCoeff(), 1e-2);
  // }

  return strategies;
}  // namespace ilqgames

}  // namespace ilqgames
