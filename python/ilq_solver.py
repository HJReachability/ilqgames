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
import matplotlib.pyplot as plt

from two_player_dynamical_system import TwoPlayerDynamicalSystem
from player_cost import PlayerCost
from solve_lq_game import solve_lq_game
from evaluate_lq_game_cost import evaluate_lq_game_cost
from visualizer import Visualizer
from logger import Logger

class ILQSolver(object):
    def __init__(self, dynamics, player1_cost, player2_cost,
                 x0, P1s, P2s, alpha1s, alpha2s,
                 u1_constraint, u2_constraint,
                 logger=None, visualizer=None):
        """
        Initialize from dynamics, player costs, current state, and initial
        guesses for control strategies for both players.

        :param dynamics: two-player dynamical system
        :type dynamics: TwoPlayerDynamicalSystem
        :param player1_cost: cost function for player 1
        :type player1_cost: PlayerCost
        :param player2_cost: cost function for player 2
        :type player2_cost: PlayerCost
        :param x0: initial state
        :type x0: np.array
        :param P1s: list of feedback gains for player 1
        :type P1s: [np.array]
        :param P2s: list of feedback gains for player 2
        :type P2s: [np.array]
        :param alpha1s: list of constant offsets for player 1
        :type alpha1s: [np.array]
        :param alpha2s: list of constant offsets for player 2
        :type alpha2s: [np.array]
        :param u1_constraint: constraint on u1
        :type u1_constraint: Constraint
        :param u2_constraint: constraint on u2
        :type u2_constraint: Constraint
        :param logger: logging utility
        :type logger: Logger
        :param visualizer: optional visualizer
        :type visualizer: Visualizer
        """
        self._dynamics = dynamics
        self._player1_cost = player1_cost
        self._player2_cost = player2_cost
        self._x0 = x0
        self._P1s = P1s
        self._P2s = P2s
        self._alpha1s = alpha1s
        self._alpha2s = alpha2s
        self._u1_constraint = u1_constraint
        self._u2_constraint = u2_constraint
        self._horizon = len(P1s)

        # Current and previous operating points (states/controls) for use
        # in checking convergence.
        self._last_operating_point = None
        self._current_operating_point = None
#        self._current_operating_point = self._compute_operating_point()

        # Fixed step size for the linesearch.
        self._alpha_scaling = 0.01

        # Set up visualizer.
        self._visualizer = visualizer
        self._logger = logger

    def run(self):
        """ Run the algorithm for the specified parameters. """
        iteration = 0

        while not self._is_converged():
            # (1) Compute current operating point and update last one.
            xs, u1s, u2s, cost1s, cost2s = self._compute_operating_point()
            self._last_operating_point = self._current_operating_point
            self._current_operating_point = (xs, u1s, u2s, cost1s, cost2s)

            if self._visualizer is not None:
                self._visualizer.add_trajectory(iteration, {
                    "xs" : xs, "u1s" : u1s, "u2s" : u2s})
                plt.clf()
                self._visualizer.plot()
                plt.pause(0.1)

            # (2) Linearize about this operating point. Make sure to
            # stack appropriately since we will concatenate state vectors
            # but not control vectors, so that
            #    ``` x_{k+1} - xs_k = A_k (x_k - xs_k) +
            #          B1_k (u1_k - u1s_k) + B2_k (u2_k - u2s_k) + c_k ```
            As = []
            B1s = []
            B2s = []
            cs = []
            for ii in range(self._horizon):
                A, B1, B2, c = self._dynamics.linearize_discrete(
                    xs[ii], u1s[ii], u2s[ii])

                As.append(A)
                B1s.append(B1)
                B2s.append(B2)
                cs.append(c)

            # (3) Quadraticize costs.
            Q1s = []; l1s = []; R11s = []; R12s = []
            Q2s = []; l2s = []; R21s = []; R22s = []
            for ii in range(self._horizon):
                # Quadraticize everything!
                _, l1, Q1, R11, R12 = self._player1_cost.quadraticize(
                    xs[ii], u1s[ii], u2s[ii])
                _, l2, Q2, R21, R22 = self._player2_cost.quadraticize(
                    xs[ii], u1s[ii], u2s[ii])

                Q1s.append(Q1)
                l1s.append(l1)
                R11s.append(R11)
                R12s.append(R12)

                Q2s.append(Q2)
                l2s.append(l2)
                R21s.append(R21)
                R22s.append(R22)

            # (4) Compute feedback Nash equilibrium of the resulting LQ game.
            P1s, P2s, alpha1s, alpha2s = solve_lq_game(
                As, B1s, B2s, cs, Q1s, Q2s, l1s, l2s, R11s, R12s, R21s, R22s)

            # Accumulate total costs for both players.
            total_cost1 = sum(cost1s).item()
            total_cost2 = sum(cost2s).item()
            print("Total cost for player 1 vs. 2: %f vs. %f." %
                  (total_cost1, total_cost2))

            # Log everything.
            if self._logger is not None:
                self._logger.log("xs", xs)
                self._logger.log("u1s", u1s)
                self._logger.log("u2s", u2s)
                self._logger.log("total_cost1", total_cost1)
                self._logger.log("total_cost2", total_cost2)
                self._logger.dump()

            # Update the member variables.
            self._P1s = P1s
            self._P2s = P2s
            self._alpha1s = alpha1s
            self._alpha2s = alpha2s

            # (5) Linesearch separately for both players.
            self._linesearch()
            iteration += 1

    def _compute_operating_point(self):
        """
        Compute current operating point by propagating through dynamics.

        :return: state and controls for both players (xs, u1s, u2s)
        :rtype: [np.array], [np.array], [np.array]
        """
        xs = [self._x0]
        u1s = []
        u2s = []
        cost1s = []
        cost2s = []

        for k in range(self._horizon):
            if self._current_operating_point is not None:
                current_x = self._current_operating_point[0][k]
                current_u1 = self._current_operating_point[1][k]
                current_u2 = self._current_operating_point[2][k]
            else:
                current_x = np.zeros((self._dynamics._x_dim, 1))
                current_u1 = np.zeros((self._dynamics._u1_dim, 1))
                current_u2 = np.zeros((self._dynamics._u2_dim, 1))

            u1 = current_u1 - self._P1s[k] @ (
                xs[k] - current_x) - self._alpha_scaling * self._alpha1s[k]
            u2 = current_u2 - self._P2s[k] @ (
                xs[k] - current_x) - self._alpha_scaling * self._alpha2s[k]

            # Clip u1 and u2.
            u1 = self._u1_constraint.clip(u1)
            u2 = self._u2_constraint.clip(u2)

            u1s.append(u1)
            u2s.append(u2)

            cost1s.append(self._player1_cost(torch.as_tensor(xs[k].copy()),
                                             torch.as_tensor(u1.copy()),
                                             torch.as_tensor(u2.copy())))
            cost2s.append(self._player2_cost(torch.as_tensor(xs[k].copy()),
                                             torch.as_tensor(u1.copy()),
                                             torch.as_tensor(u2.copy())))

            if k == self._horizon - 1:
                break

            x = self._dynamics.integrate(xs[k], u1, u2)
            xs.append(x)

        return xs, u1s, u2s, cost1s, cost2s

    def _linesearch(self):
        """ Linesearch for both players separately. """
        pass
        # HACK: This is simple, need an actual linesearch.
#        for ii in range(self._horizon):
#            self._alpha1s[ii] *= self._alpha_scaling
#            self._alpha2s[ii] *= self._alpha_scaling

    def _is_converged(self):
        """ Check if the last two operating points are close enough. """
        if self._last_operating_point is None:
            return False

        # Tolerance for comparing operating points. If all states changes
        # within this tolerance in the Euclidean norm then we've converged.
        TOLERANCE = 1e-4
        for ii in range(self._horizon):
            last_x = self._last_operating_point[0][ii]
            current_x = self._current_operating_point[0][ii]

            if np.linalg.norm(last_x - current_x) > TOLERANCE:
                return False

        return True
