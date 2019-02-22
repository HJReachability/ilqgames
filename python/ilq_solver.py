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
# Iterative LQ solver.
#
################################################################################

import numpy as np
from scipy.linalg import block_diag
import torch

from dynamical_system import DynamicalSystem
from two_player_dynamical_system import TwoPlayerDynamicalSystem
from player_cost import PlayerCost
from semiquadratic_cost import SemiquadraticCost
from semiquadratic_polyline_cost import SemiquadraticPolylineCost
from proximity_cost import ProximityCost
from solve_lq_game import solve_lq_game

from polyline import Polyline
from point import Polyline
from line_segment import LineSegment

class ILQSolver(object):
    def __init__(self, dynamics1, dynamics2, player1_cost, player2_cost,
                 x1, x2, P1s, P2s, alpha1s, alpha2s):
        """
        Initialize from dynamics, player costs, current state, and initial
        guesses for control strategies for both players.

        :param dynamics1: dynamical system for player 1
        :type dynamics1: DynamicalSystem
        :param dynamics2: dynamical system for player 2
        :type dynamics2: DynamicalSystem
        :param player1_cost: cost function for player 1
        :type player1_cost: PlayerCost
        :param player2_cost: cost function for player 2
        :type player2_cost: PlayerCost
        :param x1: initial state of player 1
        :type x1: np.array
        :param x2: initial state of player 1
        :type x2: np.array
        :param P1s: list of feedback gains for player 1
        :type P1s: [np.array]
        :param P2s: list of feedback gains for player 2
        :type P2s: [np.array]
        :param alpha1s: list of constant offsets for player 1
        :type alpha1s: [np.array]
        :param alpha2s: list of constant offsets for player 2
        :type alpha2s: [np.array]
        """
        self._dynamics1 = dynamics1
        self._dynamics2 = dynamics2
        self._player1_cost = player1_cost
        self._player2_cost = player2_cost
        self._x1 = x1
        self._x2 = x2
        self._P1s = P1s
        self._P2s = P2s
        self._alpha1s = alpha1s
        self._alpha2s = alpha2s
        self._horizon = len(P1s)

        # Current and previous operating points (states/controls) for use
        # in checking convergence.
        self._last_operating_point = None
        self._current_operating_point = self._compute_operating_point()

        # Fixed step size for the linesearch.
        self._alpha_scaling = 0.1

    def run(self):
        """ Run the algorithm for the specified parameters. """
        while not self._is_converged():
            # (1) Compute current operating point.
            x1s, u1s, x2s, u2s = self._compute_operating_point()

            # (2) Linearize about this operating point. Make sure to
            # stack appropriately since we will concatenate state vectors
            # but not control vectors, so that
            #    ``` x_{k+1}= A_k x_k + B1_k u1_k + B2_k u2_k + c_k ```
            As = []
            B1s = []
            B2s = []
            cs = []
            for ii in range(self._horizon):
                A1, B1, c1 = self._dynamics1.linearize_discrete(
                    x1s[ii], u1s[ii])
                A2, B2, c2 = self._dynamics2.linearize_discrete(
                    x2s[ii], u2s[ii])

                As.append(block_diag(A1, A2))
                B1s.append(B1)
                B2s.append(B2)
                cs.append(np.concatenate([c1, c2], axis=0))

            # (3) Quadraticize costs.
            cost1s = []; Q1s = []; l1s = []; R11s = []; R12s = []
            cost2s = []; Q2s = []; l2s = []; R21s = []; R22s = []
            for ii in range(self._horizon):
                # Concatenate state vectors.
                x = np.concatenate([x1s[ii], x2s[ii]], axis=0)

                # Quadraticize everything!
                cost1, l1, Q1, R11, R12 = self._player1_cost.quadraticize(
                    x, u1s[ii], u2s[ii])
                cost2, l2, Q2, R21, R22 = self._player2_cost.quadraticize(
                    x, u1s[ii], u2s[ii])

            # (4) Compute feedback Nash equilibrium of the resulting LQ game.
            P1s, P2s, alpha1s, alpha2s = solve_lq_game(
                As, B1s, B2s, cs, Q1s, Q2s, l1s, l2s, R11s, R12s, R21s, R22s)

            # Update the member variables.
            self._P1s = P1s
            self._P2s = P2s
            self._alpha1s = alpha1s
            self._alpha2s = alpha2s

            # (5) Linesearch separately for both players.
            self._linesearch()

    def _compute_operating_point(self):
        """
        Compute current operating point by propagating through dynamics.

        :return: states and controls for both players (x1s, u1s, x2s, u2s)
        :rtype: [np.array], [np.array], [np.array], [np.array]
        """
        x1s = [self._x1]
        u1s = []
        x2s = [self._x2]
        u2s = []

        for k in range(self._horizon):
            u1 = -self._P1s[k] @ x1s[k] - self._alpha1s[k]
            u2 = -self._P2s[k] @ x2s[k] - self._alpha2s[k]

            u1s.append(u1)
            u2s.append(u2)

            if k == self._horizon - 1:
                break

            x1 = self._dynamics1.integrate(x1s[k], u1)
            x2 = self._dynamics2.integrate(x2s[k], u2)

            x1s.append(x1)
            x2s.append(x2)

        return x1s, u1s, x2s, u2s

    def _linesearch(self):
        """ Linesearch for both players separately. """

        # HACK: This is simple, need an actual linesearch. 
        for ii in range(self._horizon):
            self._alpha1s[ii] *= self._alpha_scaling
            self._alpha2s[ii] *= self._alpha_scaling

    def _is_converged(self):
        """ Check if the last two operating points are close enough. """
        if self._last_operating_point is None:
            return False

        # Tolerance for comparing operating points. If all states changes
        # within this tolerance in the Euclidean norm then we've converged.
        TOLERANCE = 1e-4
        for ii in range(self._horizon):
            last_x1 = self._last_operating_point[0][ii]
            current_x1 = self._current_operating_point[0][ii]
            last_x2 = self._last_operating_point[2][ii]
            current_x2 = self._current_operating_point[2][ii]

            if np.linalg.norm(last_x1 - current_x1) > TOLERANCE or \
               np.linalg.norm(last_x2 - current_x2) > TOLERANCE:
                return False

        return True
