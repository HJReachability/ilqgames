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
# Function to compute the value of a two-player, time-varying, finite horizon LQ
# game for each player. Notation matches that in Definition 6.1 on pp. 269 of
# Dynamic Noncooperative Game Theory (second edition) by Tamer Basar and
# Geert Jan Olsder.
#
################################################################################

import numpy as np

def evaluate_2_player_lq_game_cost(
        As, B1s, B2s, Q1s, Q2s, l1s, l2s, R11s, R12s, R21s, R22s,
        P1s, P2s, alpha1s, alpha2s, x0):
    """
    Solve a time-varying, finite horizon LQ game (finds closed-loop Nash
    feedback strategies for both players).
    Assumes that dynamics are given by
                    x_{k+1} = A_k x_k + B1_k u1_k + B2_k u2_k + c_k

    NOTE: all of these lists of matrices must be the same length.
    NOTE: all indices correspond to the "current time" k except for those
    of Q1 and Q2, which correspond to the "next time" k+1. That is, the kth
    entry of Q1s is the state cost corresponding to time step k+1. This makes
    sense because there is no point assigning any state cost to the initial
    state x_0.
    Returns cost for both players: cost1, cost2

    :param As: A matrices (dynamics)
    :type As: [np.array]
    :param B1s: B1 matrices (dynamics)
    :type B1s: [np.array]
    :param B2s: B2 matrices (dynamics)
    :type B2s: [np.array]
    :param Q1s: state costs for player 1 (protagonist)
    :type Q1s: [np.array]
    :param Q2s: state costs for player 2 (antagonist)
    :type Q2s: [np.array]
    :param l1s: linear state costs for player 1
    :type l1s: [np.array]
    :param l2s: linear state costs for player 2
    :type l2s: [np.array]
    :param R11s: control costs for player 1
    :type R11s: [np.array]
    :param R12s: control cost for player 1 associated to player 2's control
    :type R12s: [np.array]
    :param R21s: control cost for player 2 associated to player 1's control
    :type R21s: [np.array]
    :param R22s: control costs for player 2
    :type R22s: [np.array]
    :param P1s: gain matrices for player 1
    :type P1s: [np.array]
    :param P2s: gain matrices for player 2
    :type P2s: [np.array]
    :param alpha1s: constant control offsets for player 1
    :type alpha1s: [np.array]
    :param alpha2s: constant control offsets for player 2
    :type alpha2s: [np.array]
    :param x0: initial state
    :type x0: np.array
    :return: cost for both players (cost1, cost2)
    :rtype: float, float
    """

    # Assertions to check valid input.
    assert len(As) == len(B1s) == len(B2s) == len(Q1s) == \
        len(Q2s) == len(R11s) == len(R12s) == len(R21s) == len(R22s) == \
        len(P1s) == len(P2s) == len(alpha1s) == len(alpha2s)
    horizon = len(As)

    # Accumulate cost for both players, following eq. 6.5b of Definition 6.1
    # in Basar and Olsder, pp. 269. The linear state costs may be found in
    # Remark 6.3 on pp. 281.
    x = x0.copy()
    cost1 = 0.0
    cost2 = 0.0
    for k in range(horizon):
        # Compute optimal controls.
        u1 = -P1s[k] @ x - alpha1s[k]
        u2 = -P2s[k] @ x - alpha2s[k]

        # Propagate next state.
        x = As[k] @ x + B1s[k] @ u1 + B2s[k] @ u2

        # Compute costs.
        cost1 += 0.5 * (x.T @ (Q1s[k] @ x + 2.0 * l1s[k]) + \
                        u1.T @ R11s[k] @ u1 + u2.T @ R12s[k] @ u2)
        cost2 += 0.5 * (x.T @ (Q2s[k] @ x + 2.0 * l2s[k]) + \
                        u1.T @ R21s[k] @ u1 + u2.T @ R22s[k] @ u2)

    return cost1[0, 0], cost2[0, 0]
