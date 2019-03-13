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
import matplotlib.pyplot as plt

from solve_lq_game import solve_lq_game

DT = 0.1
HORIZON = 10.0
NUM_TIMESTEPS = int(HORIZON / DT)

# 2D point mass.
A = np.eye(4) + DT * np.array([
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0]
])

B1 = DT * np.array([
    [0.0, 0.0],
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0]
])

B2 = DT * np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 0.0],
    [0.0, 0.0]
])

As = [A] * NUM_TIMESTEPS
B1s = [B1] * NUM_TIMESTEPS
B2s = [B2] * NUM_TIMESTEPS

# State costs.
Q1 = np.diag([1.0, 2.0, 0.0, 0.0]); Q1s = [Q1] * NUM_TIMESTEPS
Q2 = -Q1; Q2s = [Q2] * NUM_TIMESTEPS
#Q2 = np.array([[1.0, 0.25], [0.25, 1.0]]); Q2s = [Q2] * NUM_TIMESTEPS
l1 = np.zeros((4, 1)); l1s = [l1] * NUM_TIMESTEPS
l2 = l1; l2s = [l2] * NUM_TIMESTEPS

# Control costs.
R11 = np.eye(2); R11s = [R11] * NUM_TIMESTEPS
R12 = np.zeros((2, 2)); R12s = [R12] * NUM_TIMESTEPS
R21 = np.zeros((2, 2)); R21s = [R21] * NUM_TIMESTEPS
R22 = np.eye(2); R22s = [R22] * NUM_TIMESTEPS

# Compute Nash solution.
[P1s, P2s], [alpha1s, alpha2s] = solve_lq_game(
    As, [B1s, B2s],
    [Q1s, Q2s], [l1s, l2s], [[R11s, R12s], [R21s, R22s]])

# Compute trajectory.
x0 = np.ones((4, 1))
xs = [x0.copy()]
u1s = []
u2s = []
for kk in range(NUM_TIMESTEPS):
    u1 = -P1s[kk] @ xs[-1] - alpha1s[kk]
    u2 = -P2s[kk] @ xs[-1] - alpha2s[kk]
    x = As[kk] @ xs[-1] + B1s[kk] @ u1 + B2s[kk] @ u2
    xs.append(x)
    u1s.append(u1)
    u2s.append(u2)

plt.figure()
plt.plot([u[0, 0] for u in u1s], "*:r", label="u1")
plt.plot([u[1, 0] for u in u1s], "*:b", label="u2")
plt.legend()
plt.title("Player 1")

plt.figure()
plt.plot([u[0, 0] for u in u2s], "*:r", label="u1")
plt.plot([u[1, 0] for u in u2s], "*:b", label="u2")
plt.legend()
plt.title("Player 2")

plt.figure()
plt.plot([x[0, 0] for x in xs], "*:r", label="x1")
plt.plot([x[1, 0] for x in xs], "*:b", label="x2")
plt.legend()
plt.title("State")
plt.show()

print("done")
