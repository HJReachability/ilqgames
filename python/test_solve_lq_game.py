"""
BSD 3-Clause License

Copyright (c) 2019, HJ Reachability Group
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author(s): David Fridovich-Keil ( dfk@eecs.berkeley.edu )
"""
################################################################################
#
# Test script for solving LQ games.
#
################################################################################

import copy
import numpy as np
import unittest

from solve_lq_game import solve_lq_game
from lyap_iters_eric import coupled_DARE_solve
from evaluate_2_player_lq_game_cost import evaluate_2_player_lq_game_cost

DT = 0.1
HORIZON = 10.0
NUM_TIMESTEPS = int(HORIZON / DT)

# For a simple, interpretable test, we'll just do a 1D point mass.
# Point mass dynamics (discrete time) with both different B matrices.
# NOTE: if both B matrices are the same then testLyapunov won't pass
# unless R22 >> R11.
A = np.array([[1.0, DT], [0.0, 1.0]]); As = [A] * NUM_TIMESTEPS
B1 = np.array([[0.5 * DT * DT], [DT]]); B1s = [B1] * NUM_TIMESTEPS
#B2 = np.array([[0.5 * DT * DT], [DT]]); B2s = [B2] * NUM_TIMESTEPS
B2 = np.array([[0.32 * DT * DT], [0.11 * DT]]); B2s = [B2] * NUM_TIMESTEPS

# State costs.
Q1 = np.array([[1.0, 0.0], [0.0, 1.0]]); Q1s = [Q1] * NUM_TIMESTEPS
Q2 = -Q1; Q2s = [Q2] * NUM_TIMESTEPS
#Q2 = np.array([[1.0, 0.25], [0.25, 1.0]]); Q2s = [Q2] * NUM_TIMESTEPS
l1 = np.array([[0.0], [0.0]]); l1s = [l1] * NUM_TIMESTEPS
l2 = l1; l2s = [l2] * NUM_TIMESTEPS

# Control costs.
R11 = np.array([[1.0]]); R11s = [R11] * NUM_TIMESTEPS
R12 = np.array([[0.0]]); R12s = [R12] * NUM_TIMESTEPS
R21 = np.array([[0.0]]); R21s = [R21] * NUM_TIMESTEPS
R22 = np.array([[1.0]]); R22s = [R22] * NUM_TIMESTEPS

class TestSolveLQGame(unittest.TestCase):
    """ Tests for solving LQ games. """

    def testLyapunov(self):
        """
        For a time invariant, long horizon problem, the solution should be
        essentially the same as that found by Lyapunov iteration.
        """
        # Solve coupled DARE two different ways.
        [P1_lyap, P2_lyap], _, test = coupled_DARE_solve(
            A, B1, B2, Q1, Q2, R11, R12, R21, R22, N=1000)

        [P1s, P2s], [alpha1s, alpha2s] = solve_lq_game(
            As, [B1s, B2s],
            [Q1s, Q2s], [l1s, l2s], [[R11s, R12s], [R21s, R22s]])

        np.testing.assert_array_almost_equal(P1_lyap, P1s[0], decimal=3)
        np.testing.assert_array_almost_equal(P2_lyap, P2s[0], decimal=3)
        np.testing.assert_array_almost_equal(alpha1s[0], 0.0)
        np.testing.assert_array_almost_equal(alpha2s[0], 0.0)

    def testNashEquilibrium(self):
        """ Check that we find a Nash equilibrium. """
        # Compute Nash solution.
        [P1s, P2s], [alpha1s, alpha2s] = solve_lq_game(
            As, [B1s, B2s],
            [Q1s, Q2s], [l1s, l2s], [[R11s, R12s], [R21s, R22s]])

        # Compute optimal costs.
        x0 = np.array([[1.0], [1.0]])
        optimal_cost1, optimal_cost2 = evaluate_2_player_lq_game_cost(
            As, B1s, B2s, Q1s, Q2s, l1s, l2s, R11s, R12s, R21s, R22s,
            P1s, P2s, alpha1s, alpha2s, x0)

        # Check that random perturbations of each players' strategies (holding
        # the other player's strategy fixed) results in higher cost.
        NUM_RANDOM_PERTURBATIONS = 100
        for ii in range(NUM_RANDOM_PERTURBATIONS):
            # Copy Nash solution.
            P1s_copy = copy.deepcopy(P1s)
            P2s_copy = copy.deepcopy(P2s)
            alpha1s_copy = copy.deepcopy(alpha1s)
            alpha2s_copy = copy.deepcopy(alpha2s)

            # Perturb player 1's strategy.
            PERTURBATION_STD = 0.1
            for k in range(NUM_TIMESTEPS):
                P1s_copy[k] += PERTURBATION_STD * np.random.randn(1, 2)
                alpha1s_copy[k] += PERTURBATION_STD * np.random.randn(1, 1)

            # Compare cost for player 1.
            cost1, _ = evaluate_2_player_lq_game_cost(
                As, B1s, B2s, Q1s, Q2s, l1s, l2s, R11s, R12s, R21s, R22s,
                P1s_copy, P2s, alpha1s_copy, alpha2s, x0)
            self.assertGreaterEqual(cost1, optimal_cost1)

            # Perturb player 2's strategy.
            for k in range(NUM_TIMESTEPS):
                P2s_copy[k] += PERTURBATION_STD * np.random.randn(1, 2)
                alpha2s_copy[k] += PERTURBATION_STD * np.random.randn(1, 1)

            # Compare cost for player 2.
            _, cost2 = evaluate_2_player_lq_game_cost(
                As, B1s, B2s, Q1s, Q2s, l1s, l2s, R11s, R12s, R21s, R22s,
                P1s, P2s_copy, alpha1s, alpha2s_copy, x0)
            self.assertGreaterEqual(cost2, optimal_cost2)

if __name__ == '__main__':
    unittest.main()
