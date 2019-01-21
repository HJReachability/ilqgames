"""
BSD 3-Clause License

Copyright (c) 2018, HJ Reachability Group
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

import numpy as np
import unittest

from solve_lq_game import solve_lq_game
from lyap_iters_eric import coupled_DARE_solve

class TestSolveLQGame(unittest.TestCase):
    """ Tests for solving LQ games. """

    def testTimeInvariantLongHorizon(self):
        """
        For a time invariant, long horizon problem, the solution should be
        essentially the same as that found by Lyapunov iteration.
        """
        DT = 0.1
        HORIZON = 10.0
        NUM_TIMESTEPS = int(HORIZON / DT)

        # For a simple, interpretable test, we'll just do a 1D point mass.
        # Point mass dynamics (discrete time) with both control signals
        # affecting acceleration.
        # NOTE: could also have player 2 introduce some friction...
        A = np.array([[1.0, DT], [0.0, 1.0]]); As = [A] * NUM_TIMESTEPS
        B1 = np.array([[0.5 * DT * DT], [DT]]); B1s = [B1] * NUM_TIMESTEPS
        B2 = np.array([[0.5 * DT * DT], [DT]]); B2s = [B2] * NUM_TIMESTEPS
        c = np.array([[0.0], [0.0]]); cs = [c] * NUM_TIMESTEPS

        # State costs.
        Q1 = np.array([[1.0, 0.0], [0.0, 0.1]]); Q1s = [Q1] * NUM_TIMESTEPS
        Q2 = -Q1; Q2s = [Q2] * NUM_TIMESTEPS
        l1 = np.array([[0.0], [0.0]]); l1s = [l1] * NUM_TIMESTEPS
        l2 = l1; l2s = [l2] * NUM_TIMESTEPS

        # Control costs.
        R11 = np.array([[1.0]]); R11s = [R11] * NUM_TIMESTEPS
        R12 = np.array([[0.0]]); R12s = [R12] * NUM_TIMESTEPS
        R21 = np.array([[0.0]]); R21s = [R21] * NUM_TIMESTEPS
        R22 = np.array([[1.0]]); R22s = [R22] * NUM_TIMESTEPS

        # Solve coupled DARE two different ways.
        [P1_lyap, P2_lyap], _, test = coupled_DARE_solve(
            A, B1, B2, Q1, Q2, R11, R12, R21, R22, N=1000)

        P1s, P2s, alpha1s, alpha2s = solve_lq_game(
            As, B1s, B2s, cs, Q1s, Q2s, l1s, l2s, R11s, R12s, R21s, R22s)

        print("Lyapunov iterations: ", P1_lyap, " / ", P2_lyap, " test: ", test)
        print("Time varying Ps: ", P1s[0], " / ", P2s[0])
        print("Time varying alphas: ", alpha1s[0], " / ", alpha2s[0])

        np.testing.assert_array_almost_equal(P1_lyap, P1s[0], decimal=4)
        np.testing.assert_array_almost_equal(P2_lyap, P2s[0], decimal=4)
        np.testing.assert_array_almost_equal(alpha1s[0], 0.0)
        np.testing.assert_array_almost_equal(alpha2s[0], 0.0)

if __name__ == '__main__':
    unittest.main()
