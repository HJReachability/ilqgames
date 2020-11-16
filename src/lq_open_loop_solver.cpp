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
#include <Eigen/Core>
#include <vector>

namespace ilqgames {

std::vector<Strategy> LQOpenLoopSolver::Solve(
    const std::vector<LinearDynamicsApproximation>& linearization,
    const std::vector<std::vector<QuadraticCostApproximation>>&
        quadraticization,
    const VectorXf& x0, std::vector<VectorXf>* delta_xs,
    std::vector<std::vector<VectorXf>>* costates) {
  CHECK_EQ(linearization.size(), num_time_steps_);
  CHECK_EQ(quadraticization.size(), num_time_steps_);

  // Make sure delta_xs and costates are the right size.
  if (delta_xs) CHECK_NOTNULL(costates);
  if (costates) CHECK_NOTNULL(delta_xs);
  if (delta_xs) {
    delta_xs->resize(num_time_steps_);
    costates->resize(num_time_steps_);
    for (size_t kk = 0; kk < num_time_steps_; kk++) {
      (*delta_xs)[kk].resize(dynamics_->XDim());
      (*costates)[kk].resize(dynamics_->NumPlayers());
      for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++)
        (*costates)[kk][ii].resize(dynamics_->XDim());
    }
  }

  // List of player-indexed strategies (each of which is a time-indexed
  // affine state error-feedback controller). Since this is an open-loop
  // strategy, we will not change the default zero value of the P matrix.
  std::vector<Strategy> strategies;
  for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++)
    strategies.emplace_back(num_time_steps_, dynamics_->XDim(),
                            dynamics_->UDim(ii));

  // Initialize m^i and M^i and index first by time and then by player.
  for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++) {
    ms_.back()[ii] = quadraticization.back()[ii].state.grad;
    Ms_.back()[ii] = quadraticization.back()[ii].state.hess;
  }

  // (1) Work backward in time and cache "special" terms.
  // NOTE: time starts from the second-to-last entry since we'll treat the
  // final entry as a terminal cost as in Basar and Olsder, ch. 6.
  for (int kk = num_time_steps_ - 2; kk >= 0; kk--) {
    // Unpack linearization and quadraticization at this time step.
    const auto& lin = linearization[kk];
    const auto& quad = quadraticization[kk];

    // Campute capital lambdas.
    capital_lambdas_[kk].setIdentity();
    for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++) {
      const auto control_iter = quad[ii].control.find(ii);
      CHECK(control_iter != quad[ii].control.end());

      chol_Rs_[kk][ii].compute(control_iter->second.hess);
      warped_Bs_[kk][ii] = chol_Rs_[kk][ii].solve(lin.Bs[ii].transpose());
      warped_rs_[kk][ii] = chol_Rs_[kk][ii].solve(control_iter->second.grad);
      capital_lambdas_[kk] += lin.Bs[ii] * warped_Bs_[kk][ii] * Ms_[kk + 1][ii];
    }

    // Compute inv(capital lambda).
    qr_capital_lambdas_[kk].compute(capital_lambdas_[kk]);

    // Compute Ms and ms.
    intermediate_terms_[kk].setZero();
    for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++) {
      intermediate_terms_[kk] -=
          lin.Bs[ii] *
          (warped_Bs_[kk][ii] * ms_[kk + 1][ii] + warped_rs_[kk][ii]);
    }

    for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++) {
      Ms_[kk][ii] =
          quad[ii].state.hess + lin.A.transpose() * Ms_[kk + 1][ii] *
                                    qr_capital_lambdas_[kk].solve(lin.A);
      ms_[kk][ii] =
          quad[ii].state.grad +
          lin.A.transpose() * (ms_[kk + 1][ii] +
                               Ms_[kk + 1][ii] * qr_capital_lambdas_[kk].solve(
                                                     intermediate_terms_[kk]));
    }
  }

  // (2) Now compute optimal state and control trajectory forward in time.
  VectorXf x_star = x0;
  VectorXf last_x_star;
  for (size_t kk = 0; kk < num_time_steps_ - 1; kk++) {
    // Maybe set delta_x.
    if (delta_xs) (*delta_xs)[kk] = x_star;

    // Unpack linearization at this time step.
    const auto& lin = linearization[kk];

    // Compute optimal x.
    last_x_star = x_star;
    x_star = qr_capital_lambdas_[kk].solve(lin.A * last_x_star +
                                           intermediate_terms_[kk]);

    // Compute optimal u and store (sign flipped) in alpha.
    // Also maybe compute costates.
    for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++) {
      const VectorXf intermediate_term =
          Ms_[kk + 1][ii] * x_star + ms_[kk + 1][ii];
      strategies[ii].alphas[kk] =
          warped_Bs_[kk][ii] * intermediate_term + warped_rs_[kk][ii];

      if (costates) (*costates)[kk][ii] = lin.A.transpose() * intermediate_term;
    }

    // Check dynamic feasibility.
    // VectorXf check_x = lin.A * last_x_star;
    // for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++)
    //   check_x -= lin.Bs[ii] * strategies[ii].alphas[kk];

    // CHECK_LE((x_star - check_x).cwiseAbs().maxCoeff(), 1e-1);
  }

  // Set delta_x and costate for last time step.
  if (delta_xs) {
    delta_xs->back() = x_star;
    for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++)
      costates->back()[ii].setZero();
  }

  return strategies;
}

}  // namespace ilqgames
