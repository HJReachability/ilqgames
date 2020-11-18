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

#ifndef ILQGAMES_SOLVER_LQ_OPEN_LOOP_SOLVER_H
#define ILQGAMES_SOLVER_LQ_OPEN_LOOP_SOLVER_H

#include <ilqgames/dynamics/multi_player_integrable_system.h>
#include <ilqgames/solver/lq_solver.h>
#include <ilqgames/utils/linear_dynamics_approximation.h>
#include <ilqgames/utils/quadratic_cost_approximation.h>
#include <ilqgames/utils/strategy.h>
#include <vector>

namespace ilqgames {

class LQOpenLoopSolver : public LQSolver {
 public:
  ~LQOpenLoopSolver() {}
  LQOpenLoopSolver(
      const std::shared_ptr<const MultiPlayerIntegrableSystem>& dynamics,
      size_t num_time_steps)
      : LQSolver(dynamics, num_time_steps) {
    // Initialize Ms and ms.
    Ms_.resize(num_time_steps_);
    ms_.resize(num_time_steps_);
    for (size_t kk = 0; kk < num_time_steps_; kk++) {
      Ms_[kk].resize(dynamics_->NumPlayers(),
                     MatrixXf::Zero(dynamics_->XDim(), dynamics_->XDim()));
      ms_[kk].resize(dynamics_->NumPlayers(),
                     VectorXf::Zero(dynamics_->XDim()));
    }

    // Initialize other "special" terms and decompositions.
    intermediate_terms_.resize(num_time_steps_ - 1,
                               VectorXf::Zero(dynamics_->XDim()));
    capital_lambdas_.resize(
        num_time_steps_ - 1,
        MatrixXf::Zero(dynamics_->XDim(), dynamics_->XDim()));
    qr_capital_lambdas_.resize(
        num_time_steps_ - 1,
        Eigen::HouseholderQR<MatrixXf>(dynamics_->XDim(), dynamics_->XDim()));

    std::vector<Eigen::LDLT<MatrixXf>> chol_Rs_element;
    std::vector<MatrixXf> warped_Bs_element;
    std::vector<VectorXf> warped_rs_element;
    for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++) {
      chol_Rs_element.emplace_back(dynamics_->UDim(ii));
      warped_Bs_element.emplace_back(dynamics_->UDim(ii), dynamics_->XDim());
      warped_rs_element.emplace_back(dynamics_->UDim(ii));
    }

    chol_Rs_.resize(num_time_steps_ - 1, chol_Rs_element);
    warped_Bs_.resize(num_time_steps_ - 1, warped_Bs_element);
    warped_rs_.resize(num_time_steps_ - 1, warped_rs_element);
  }

  // Solve underlying LQ game to a open-loop Nash equilibrium.
  // Optionally return delta xs and costates.
  std::vector<Strategy> Solve(
      const std::vector<LinearDynamicsApproximation>& linearization,
      const std::vector<std::vector<QuadraticCostApproximation>>&
          quadraticization,
      const VectorXf& x0, std::vector<VectorXf>* delta_xs = nullptr,
      std::vector<std::vector<VectorXf>>* costates = nullptr);

 private:
  // Initialize Ms and ms.
  std::vector<std::vector<VectorXf>> ms_;
  std::vector<std::vector<MatrixXf>> Ms_;

  // Instantiate the rest of the "special" terms and decompositions.
  std::vector<VectorXf> intermediate_terms_;
  std::vector<MatrixXf> capital_lambdas_;
  std::vector<Eigen::HouseholderQR<MatrixXf>> qr_capital_lambdas_;
  std::vector<std::vector<Eigen::LDLT<MatrixXf>>> chol_Rs_;
  std::vector<std::vector<MatrixXf>> warped_Bs_;
  std::vector<std::vector<VectorXf>> warped_rs_;

};  // LQOpenLoopSolver

}  // namespace ilqgames

#endif
