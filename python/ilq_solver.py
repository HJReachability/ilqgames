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
        # TODO!
        pass

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
        """ Compute current operating point by propagating through dynamics. """
        pass

    def _linesearch(self):
        """ Linesearch for both players separately. """
        pass
