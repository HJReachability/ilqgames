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
# Function to solve time-varying finite horizon LQ game.
# Please refer to Corollary 6.1 on pp. 279 of Dynamic Noncooperative Game Theory
# (second edition) by Tamer Basar and Geert Jan Olsder.
#
################################################################################

import numpy as np
from collections import deque

def solve_lq_game(As, B1s, B2s, cs, Q1s, Q2s, l1s, l2s, R11s, R12s, R21s, R22s):
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
    Returns P1s, P2s, alpha1s, alpha2s

    :param As: A matrices (dynamics)
    :type As: [np.array]
    :param B1s: B1 matrices (dynamics)
    :type B1s: [np.array]
    :param B2s: B2 matrices (dynamics)
    :type B2s: [np.array]
    :param cs: drift terms (dynamics)
    :type cs: [np.array]
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
    :return: gain matrices and constant offsets (ui_k = -Pi_k - alphai_k)
       for both players (P1s, P2s, alpha1s, alpha2s)
    :rtype: [np.array], [np.array], [np.array], [np.array]
    """

    # Assertions to check valid input.
    assert len(As) == len(B1s) == len(B2s) == len(cs) == len(Q1s) == \
        len(Q2s) == len(R11s) == len(R12s) == len(R21s) == len(R22s)
    horizon = len(As) - 1

    # Cache dimensions of control and state.
    u1_dim = B1s[0].shape[1]
    x_dim = As[0].shape[0]

    # Note: notation and variable naming closely follows that introduced in
    # the "Preliminary Notation for Corollary 6.1" section, which may be found
    # on pp. 279 of Basar and Olsder.

    # Recursive computation of all intermediate and final variables.
    # Use deques for efficient prepending.
    Z1s = deque([Q1s[-1]])
    Z2s = deque([Q2s[-1]])
    zeta1s = deque([l1s[-1]])
    zeta2s = deque([l2s[-1]])
    Fs = deque()
    P1s = deque()
    P2s = deque()
    betas = deque()
    alpha1s = deque()
    alpha2s = deque()
    for k in range(horizon, -1, -1):
        # Unpack all relevant variables.
        A = As[k]
        B1 = B1s[k]
        B2 = B2s[k]
        c = cs[k]
        Q1 = Q1s[k]
        Q2 = Q2s[k]
        l1 = l1s[k]
        l2 = l2s[k]
        R11 = R11s[k]
        R12 = R12s[k]
        R21 = R21s[k]
        R22 = R22s[k]
        Z1 = Z1s[0]
        Z2 = Z2s[0]
        zeta1 = zeta1s[0]
        zeta2 = zeta2s[0]

        # Compute P1 and P2 given previously computed Z1 and Z2.
        # Refer to equation 6.17a in Basar and Olsder.
        # This will involve solving a system of matrix linear equations of the
        # form [S11, S12; S21, S22] * [P1; P2] = [Y1; Y2].
        S11 = R11 + B1.T @ Z1 @ B1
        S12 = B1.T @ Z1 @ B2
        S21 = B2.T @ Z2 @ B1
        S22 = R22 + B2.T @ Z2 @ B2
        S_top = np.concatenate([S11, S12], axis=1)
        S_bot = np.concatenate([S21, S22], axis=1)
        S = np.concatenate([S_top, S_bot], axis=0)

        Y1 = B1.T @ Z1 @ A
        Y2 = B2.T @ Z2 @ A
        Y = np.concatenate([Y1, Y2], axis=0)

        P = np.linalg.solve(a=S, b=Y)
        P1 = P[:u1_dim, :]
        P2 = P[u1_dim:, :]
        P1s.appendleft(P1)
        P2s.appendleft(P2)

        # Compute F_k = A_k - B1_k P1_k - B2_k P2_k.
        # This is eq. 6.17c from Basar and Olsder.
        F = A - B1 @ P1 - B2 @ P2
        Fs.appendleft(F)

        # Update Z1 and Z2 to be the next step earlier in time (now they
        # correspond to time k+1).
        Z1s.appendleft(F.T @ Z1 @ F + Q1 + P1.T @ R11 @ P1 + P2.T @ R12 @ P2)
        Z2s.appendleft(F.T @ Z2 @ F + Q2 + P1.T @ R21 @ P1 + P2.T @ R21 @ P2)

        # Compute alpha1 and alpha2 using previously computed zeta1 and zeta2.
        # Refer to equation 6.17d in Basar and Olsder.
        # This will involve solving a system of linear matrix equations of the
        # form [S11, S12; S21, S22] * [alpha1; alpha2] = [Y1; Y2].
        S11 = R11 + B1.T @ Z1 @ B1
        S12 = B1.T @ Z1 @ B2
        S21 = B2.T @ Z2 @ B1
        S22 = R22 + B2.T @ Z2 @ B2
        S_top = np.concatenate([S11, S12], axis=1)
        S_bot = np.concatenate([S21, S22], axis=1)
        S = np.concatenate([S_top, S_bot], axis=0)

        Y1 = B1.T @ (zeta1 + Z1 @ c)
        Y2 = B2.T @ (zeta2 + Z2 @ c)
        Y = np.concatenate([Y1, Y2], axis=0)

        alpha = np.linalg.solve(a=S, b=Y)
        alpha1 = alpha[:u1_dim]
        alpha2 = alpha[u1_dim:]
        alpha1s.appendleft(alpha1)
        alpha2s.appendleft(alpha2)

        # Compute beta_k = c_k - B1_k alpha1 - B2_k alpha2_k.
        # This is eq. 6.17f in Basar and Olsder.
        # NOTE: in eq. 6.17f, both B1 and B2 are transposed, but I believe
        # that this is a typo.
        beta = c - B1 @ alpha1 - B2 @ alpha2
        betas.appendleft(beta)

        # Update zeta1 and zeta2 to be the next step earlier in time (now they
        # correspond to time k+1).
        zeta1s.appendleft(F.T @ (zeta1 + Z1 @ beta) + l1 +
                          P1.T @ R11 @ alpha1 + P2.T @ R12 @ alpha2)
        zeta2s.appendleft(F.T @ (zeta2 + Z2 @ beta) + l2 +
                          P1.T @ R21 @ alpha1 + P2.T @ R22 @ alpha2)

    # Return P1s, P2s, alpha1s, alpha2s
    return list(P1s), list(P2s), list(alpha1s), list(alpha2s)
