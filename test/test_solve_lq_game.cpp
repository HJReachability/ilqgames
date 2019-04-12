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
// Test SolveLQGame by comparing to an implementation of Lyapunov iterations
// and ensuring that the solutions agree on a two-player time-invariant
// long-horizon example.
//
// Adapted from original Python implementation of Lyapunov iterations written
// by Eric Mazumdar (2018). Algorithm may be found at:
// https://link.springer.com/chapter/10.1007/978-1-4612-4274-1_17
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/solver/solve_lq_game.h>
#include <ilqgames/utils/types.h>

#include <gtest/gtest.h>
#include <math.h>

using namespace ilqgames;

namespace {
// Solve two-player infinite horizon (time-invariant) LQ game by Lyapunov
// iterations.
void SolveLyapunovIterations(const MatrixXf& A, const MatrixXf& B1,
                             const MatrixXf& B2, const MatrixXf& Q1,
                             const MatrixXf& Q2, const MatrixXf& R11,
                             const MatrixXf& R12, const MatrixXf& R21,
                             const MatrixXf& R22, Eigen::Ref<MatrixXf> P1,
                             Eigen::Ref<MatrixXf> P2) {
  ASSERT_TRUE(P1.get() && P2.get());

  // Number of iterations.
  constexpr size_t kNumIterations = 100;

  // Initialize Zs to Qs.
  MatrixXf Z1 = Q1;
  MatrixXf Z2 = Q2;

  // Initialize Ps.
  P1 = (R11 + B1.transpose() * Z1 * B1)
           .householderQr()
           .solve(B1.transpose() * Z1 * A);
  P2 = (R22 + B2.transpose() * Z2 * B2)
           .householderQr()
           .solve(B2.transpose() * Z2 * A);

  for (size_t ii = 0; ii < kNumIterations; ii++) {
    const MatrixXf old_P1 = P1;
    const MatrixXf old_P2 = P2;

    P1 = (R11 + B1.transpose() * Z1 * B1)
             .householderQr()
             .solve(B1.transpose() * Z1 * (A - B2 * old_P2));
    P2 = (R22 + B2.transpose() * Z2 * B2)
             .householderQr()
             .solve(B2.transpose() * Z2 * (A - B1 * old_P1));

    Z1 = (A - B1 * P1 - B2 * P2).transpose() * Z1 * (A - B1 * P1 - B2 * P2) +
         P1.transpose() * R11 * P1 + P2.transpose() * R12 * P2 + Q1;
    Z2 = (A - B1 * P1 - B2 * P2).transpose() * Z2 * (A - B1 * P1 - B2 * P2) +
         P1.transpose() * R21 * P1 + P2.transpose() * R22 * P2 + Q2;
  }
}
}  // anonymous namespace

TEST(SolveLQGameTest, MatchesLyapunovIterations) {
  // Construct a 2-player time-invariant LQ game.

  // Solve with Lyapunov iterations.

  // Solve with the general (time-varying, finite-horizon) solver.

  // Check that the answers are close.
}
