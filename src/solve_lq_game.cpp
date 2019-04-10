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
//
// Solve a time-varying, finite horizon LQ game (finds closed-loop Nash
// feedback strategies for both players).
//
// Assumes that dynamics are given by
//           ``` dx_{k+1} = A_k dx_k + \sum_i Bs[i]_k du[i]_k ```
//
// NOTE: Bs, Qs, ls, R1s, R2s are all lists of lists of matrices.
// NOTE: all indices of inner lists correspond to the "current time" k except
// for those of the Qs, which correspond to the "next time" k+1. That is,
// the kth entry of Qs[i] is the state cost corresponding to time step k+1. This
// makes sense because there is no point assigning any state cost to the
// initial state x_0.
//
// Returns strategies Ps, alphas.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/dynamics/multi_player_dynamical_system.h>
#include <ilqgames/solver/solve_lq_game.h>
#include <ilqgames/utils/linear_dynamics_approximation.h>
#include <ilqgames/utils/quadratic_cost_approximation.h>
#include <ilqgames/utils/strategy.h>

#include <vector>

namespace ilqgames {

std::vector<Strategy> SolveLQGame(
    const MultiPlayerDynamicalSystem& dynamics,
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
  for (size_t ii = 0; ii < dynamics.NumPlayers(); ii++)
    strategies.emplace_back(horizon, dynamics.XDim(), dynamics.UDim(ii));

  // Cache the total number of control dimensions, since this is inefficient
  // to compute.
  const Dimension total_udim = dynamics.TotalUDim();

  // Quadratic components of value function at the current time step in the
  // dynamic program.
  // NOTE: since these will be computed by solving a big linear matrix equation
  // S [Ps, alphas] = [YPs, Yalphas] (i.e., S X = Y), we will pre-allocate the
  // memory for that equation and define these components as Eigen::Refs.
  MatrixXf S(total_udim, total_udim);
  MatrixXf X(total_udim, dynamics.XDim() + 1);
  MatrixXf Y(total_udim, dynamics.XDim() + 1);

  std::vector<Eigen::Ref<MatrixXf>> Zs;
  std::vector<Eigen::Ref<VectorXf>> zetas;
  Dimension cumulative_udim = 0;
  for (PlayerIndex ii = 0; ii < dynamics.NumPlayers(); ii++) {
    Zs.push_back(
        X.block(cumulative_udim, 0, dynamics.UDim(ii), dynamics.XDim()));
    zetas.push_back(
        X.col(dynamics.XDim()).segment(cumulative_udim, dynamics.UDim(ii)));
    cumulative_udim += dynamics.UDim(ii);
  }

  return strategies;
}

}  // namespace ilqgames
