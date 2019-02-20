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
import torch

from player_cost import PlayerCost
from semiquadratic_cost import SemiquadraticCost
from semiquadratic_polyline_cost import SemiquadraticPolylineCost
from proximity_cost import ProximityCost

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

    def run(self):
        """ Run the algorithm for the specified parameters. """
        # TODO: loop out here.
        # (1) Compute current operating point.
        # (2) Linearize about this operating point.
        # (3) Quadraticize costs.
        # (4) Compute feedback Nash equilibrium of the resulting LQ game.
        # (5) Linesearch separately for both players.
        # TODO: figure out if this is actually the right thing to do here.
        pass

    def _compute_operating_point(self):
        """ Compute current operating point by propagating through dynamics. 
        
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
        pass
