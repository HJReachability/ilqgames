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
// Core LQ game solver from Basar and Olsder, "Preliminary Notation for
// Corollary 6.1" (pp. 279). All notation matches the text, though we
// shall assume that `c` (additive drift in dynamics) is always `0`, which
// holds because these dynamics are for delta x, delta us.
// Also, we have modified terms slightly to account for linear terms in the
// stage cost for control, i.e.
//       control penalty i = 0.5 \sum_j du_j^T R_ij (du_j + 2 r_ij)
//
// Solve a time-varying, finite horizon LQ game (finds closed-loop Nash
// feedback strategies for both players).
//
// Assumes that dynamics are given by
//           ``` dx_{k+1} = A_k dx_k + \sum_i Bs[i]_k du[i]_k ```
//
// Returns strategies Ps, alphas.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_SOLVER_LQ_FEEDBACK_SOLVER_H
#define ILQGAMES_SOLVER_LQ_FEEDBACK_SOLVER_H

#include <ilqgames/dynamics/multi_player_integrable_system.h>
#include <ilqgames/solver/lq_solver.h>
#include <ilqgames/utils/linear_dynamics_approximation.h>
#include <ilqgames/utils/quadratic_cost_approximation.h>
#include <ilqgames/utils/strategy.h>

#include <vector>

namespace ilqgames {

class LQFeedbackSolver : public LQSolver {
 public:
  ~LQFeedbackSolver() {}
  LQFeedbackSolver(
      const std::shared_ptr<const MultiPlayerIntegrableSystem>& dynamics,
      size_t num_time_steps, bool adaptive_regularization = true)
      : LQSolver(dynamics, num_time_steps),
        adaptive_regularization_(adaptive_regularization) {
    // Cache the total number of control dimensions, since this is inefficient
    // to compute.
    const Dimension total_udim = dynamics_->TotalUDim();  // 2

    // Preallocate memory for coupled Riccati solve at each time step and make
    // Eigen::Refs to the solution.
    S_.resize(total_udim, total_udim);
    X_.resize(total_udim, dynamics_->XDim() + 1);
    Y_.resize(total_udim, dynamics_->XDim() + 1);

    Dimension cumulative_udim = 0;
    for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++) {
      Ps_.push_back(
          X_.block(cumulative_udim, 0, dynamics_->UDim(ii), dynamics_->XDim()));
      alphas_.push_back(X_.col(dynamics_->XDim())
                            .segment(cumulative_udim, dynamics_->UDim(ii)));

      // Increment cumulative_udim.
      cumulative_udim += dynamics_->UDim(ii);
    }

    // Initialize Zs and zetas for each time and player. Note that we need to
    // store over all time to compute optimal costates if desired.
    Zs_.resize(num_time_steps_);
    zetas_.resize(num_time_steps_);
    for (size_t kk = 0; kk < num_time_steps_; kk++) {
      Zs_[kk].resize(dynamics_->NumPlayers());
      zetas_[kk].resize(dynamics_->NumPlayers());
      for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++) {
        Zs_[kk][ii].resize(dynamics_->XDim(), dynamics_->XDim());
        zetas_[kk][ii].resize(dynamics_->XDim());
      }
    }

    // Preallocate memory for intermediate variables F, beta.
    F_.resize(dynamics_->XDim(), dynamics_->XDim());
    beta_.resize(dynamics_->XDim());
  }

  // Solve underlying LQ game to a feedback Nash equilibrium.
  // Optionally return delta xs and costates.
  std::vector<Strategy> Solve(
      const std::vector<LinearDynamicsApproximation>& linearization,
      const std::vector<std::vector<QuadraticCostApproximation>>&
          quadraticization,
      const VectorXf& x0, std::vector<VectorXf>* delta_xs = nullptr,
      std::vector<std::vector<VectorXf>>* costates = nullptr);

 private:
  // Quadratic/linear components of value function at the current time step in
  // the dynamic program.
  // NOTE: since these will be computed by solving a big
  // linear matrix equation S [Ps, alphas] = [YPs, Yalphas] (i.e., S X = Y), we
  // will pre-allocate the memory for that equation and define these components
  // as Eigen::Refs.
  MatrixXf S_, X_, Y_;
  std::vector<Eigen::Ref<MatrixXf>> Ps_;
  std::vector<Eigen::Ref<VectorXf>> alphas_;

  // Initialize Zs and zetas for each time and player.
  std::vector<std::vector<MatrixXf>> Zs_;
  std::vector<std::vector<VectorXf>> zetas_;

  // Preallocate memory for intermediate variables F, beta.
  MatrixXf F_;
  VectorXf beta_;

  // Adaptive regularization using Gershgorin circle theorem.
  const bool adaptive_regularization_;
};  // LQFeedbackSolver

}  // namespace ilqgames

#endif
