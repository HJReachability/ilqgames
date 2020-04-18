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
// Core open-loop LQ game solver from Basar and Olsder, Chapter 6. All notation
// matches the text, though we shall assume that `c` (additive drift in
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
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/dynamics/multi_player_integrable_system.h>
#include <ilqgames/solver/lq_feedback_solver.h>
#include <ilqgames/solver/lq_open_loop_solver.h>
#include <ilqgames/utils/linear_dynamics_approximation.h>
#include <ilqgames/utils/quadratic_cost_approximation.h>
#include <ilqgames/utils/strategy.h>

#include <glog/logging.h>
#include <vector>

namespace ilqgames {

std::vector<Strategy> LQOpenLoopSolver::Solve(
    const MultiPlayerIntegrableSystem& dynamics,
    const std::vector<LinearDynamicsApproximation>& linearization,
    const std::vector<std::vector<QuadraticCostApproximation>>&
        quadraticization) {
  // Unpack horizon.
  const size_t horizon = linearization.size();
  CHECK_EQ(quadraticization.size(), horizon);
  CHECK_GT(horizon, 0);

  // List of player-indexed strategies (each of which is a time-indexed
  // affine state error-feedback controller).
  std::vector<Strategy> strategies;
  for (PlayerIndex ii = 0; ii < dynamics.NumPlayers(); ii++)
    strategies.emplace_back(horizon, dynamics.XDim(), dynamics.UDim(ii));

  // Cache the total number of control dimensions, since this is inefficient
  // to compute.
  const Dimension total_udim = dynamics.TotalUDim();

  // Quadratic/linear components of value function at the current time step in
  // the dynamic program.
  // NOTE: since these will be computed by solving a big
  // linear matrix equation S [Ps, alphas] = [YPs, Yalphas] (i.e., S X = Y), we
  // will pre-allocate the memory for that equation and define these components
  // as Eigen::Refs.
  MatrixXf S(total_udim, total_udim);
  MatrixXf X(total_udim, dynamics.XDim() + 1);
  MatrixXf Y(total_udim, dynamics.XDim() + 1);

  std::vector<Eigen::Ref<MatrixXf>> Ps;
  std::vector<Eigen::Ref<VectorXf>> alphas;
  Dimension cumulative_udim = 0;
  for (PlayerIndex ii = 0; ii < dynamics.NumPlayers(); ii++) {
    Ps.push_back(
        X.block(cumulative_udim, 0, dynamics.UDim(ii), dynamics.XDim()));
    alphas.push_back(
        X.col(dynamics.XDim()).segment(cumulative_udim, dynamics.UDim(ii)));

    // Increment cumulative_udim.
    cumulative_udim += dynamics.UDim(ii);
  }

  // Initialize Zs and zetas at the final time.
  std::vector<MatrixXf> Zs(dynamics.NumPlayers());
  std::vector<VectorXf> zetas(dynamics.NumPlayers());
  for (PlayerIndex ii = 0; ii < dynamics.NumPlayers(); ii++) {
    Zs[ii] = quadraticization.back()[ii].state.hess;
    zetas[ii] = quadraticization.back()[ii].state.grad;
  }

  // Preallocate memory for intermediate variables F, beta.
  MatrixXf F(dynamics.XDim(), dynamics.XDim());
  VectorXf beta(dynamics.XDim());

  // Work backward in time and solve the dynamic program.
  // NOTE: time starts from the second-to-last entry since we'll treat the final
  // entry as a terminal cost as in Basar and Olsder, ch. 6.
  for (int kk = horizon - 2; kk >= 0; kk--) {
    // Unpack linearization and quadraticization at this time step.
    const auto& lin = linearization[kk];
    const auto& quad = quadraticization[kk];

    // Populate coupling matrix S for linear matrix equation to determine X (Ps
    // and alphas).
    // NOTE: S is generally dense and asymmetric, though it is symmetric if all
    // players have the same Z.
    Dimension cumulative_udim_row = 0;
    for (PlayerIndex ii = 0; ii < dynamics.NumPlayers(); ii++) {
      // Intermediate variable to store B[ii]' * Z[ii].
      const MatrixXf BiZi = lin.Bs[ii].transpose() * Zs[ii];

      Dimension cumulative_udim_col = 0;
      for (PlayerIndex jj = 0; jj < dynamics.NumPlayers(); jj++) {
        Eigen::Ref<MatrixXf> S_block =
            S.block(cumulative_udim_row, cumulative_udim_col, dynamics.UDim(ii),
                    dynamics.UDim(jj));

        if (ii == jj) {
          // Does player ii's cost depend upon player jj's control?
          const auto control_iter = quad[ii].control.find(ii);
          CHECK(control_iter != quad[ii].control.end());

          S_block = BiZi * lin.Bs[ii] + control_iter->second.hess;
        } else {
          S_block = BiZi * lin.Bs[jj];
        }

        // Increment cumulative_udim_col.
        cumulative_udim_col += dynamics.UDim(jj);
      }

      // Set appropriate blocks of Y.
      Y.block(cumulative_udim_row, 0, dynamics.UDim(ii), dynamics.XDim()) =
          BiZi * lin.A;
      Y.col(dynamics.XDim()).segment(cumulative_udim_row, dynamics.UDim(ii)) =
          lin.Bs[ii].transpose() * zetas[ii] + quad[ii].control.at(ii).grad;

      // Increment cumulative_udim_row.
      cumulative_udim_row += dynamics.UDim(ii);
    }

    // Solve linear matrix equality S X = Y.
    // NOTE: not 100% sure that this avoids dynamic memory allocation.
    X = S.householderQr().solve(Y);

    // Set strategy at current time step.
    for (PlayerIndex ii = 0; ii < dynamics.NumPlayers(); ii++) {
      strategies[ii].Ps[kk] = Ps[ii];
      strategies[ii].alphas[kk] = alphas[ii];
    }

    // Compute F and beta.
    F = lin.A;
    beta = VectorXf::Zero(dynamics.XDim());
    for (PlayerIndex ii = 0; ii < dynamics.NumPlayers(); ii++) {
      F -= lin.Bs[ii] * Ps[ii];
      beta -= lin.Bs[ii] * alphas[ii];
    }

    // Update Zs and zetas.
    for (PlayerIndex ii = 0; ii < dynamics.NumPlayers(); ii++) {
      zetas[ii] =
          (F.transpose() * (zetas[ii] + Zs[ii] * beta) + quad[ii].state.grad)
              .eval();
      Zs[ii] = (F.transpose() * Zs[ii] * F + quad[ii].state.hess).eval();

      // Add terms for nonzero Rijs.
      for (const auto& Rij_entry : quad[ii].control) {
        const PlayerIndex jj = Rij_entry.first;
        const MatrixXf& Rij = Rij_entry.second.hess;
        const VectorXf& rij = Rij_entry.second.grad;
        zetas[ii] += Ps[jj].transpose() * (Rij * alphas[jj] - rij);
        Zs[ii] += Ps[jj].transpose() * Rij * Ps[jj];
      }
    }
  }

  return strategies;
}

}  // namespace ilqgames
