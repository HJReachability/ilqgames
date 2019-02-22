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
# 4D unicycle model with disturbance. Dynamics are as follows:
#                          \dot x     = v cos theta + u21
#                          \dot y     = v sin theta + u22
#                          \dot theta = u11
#                          \dot v     = u12
#
################################################################################

import torch
import numpy as np

from two_player_dynamical_system import TwoPlayerDynamicalSystem

class TwoPlayerUnicycle4D(TwoPlayerDynamicalSystem):
    """ 4D unicycle model with disturbances. """

    def __init__(self, T=0.1):
        super(TwoPlayerUnicycle4D, self).__init__(4, 2, 2, T)

    def __call__(self, x, u1, u2):
        """
        Compute the time derivative of state for a particular state/control.
        NOTE: `x`, `u1`, and `u2` should be 2D (i.e. column vectors).

        :param x: current state
        :type x: torch.Tensor or np.array
        :param u1: current control input
        :type u1: torch.Tensor or np.array
        :param u2: current disturbance input
        :type u2: torch.Tensor or np.array
        :return: current time derivative of state
        :rtype: torch.Tensor or np.array
        """
        if isinstance(x, np.ndarray):
            assert isinstance(u1, np.ndarray)
            assert isinstance(u2, np.ndarray)
            x_dot = np.zeros((self._x_dim, 1))
            cos = np.cos
            sin = np.sin
        else:
            assert isinstance(u1, torch.Tensor)
            assert isinstance(u2, torch.Tensor)
            x_dot = torch.zeros((self._x_dim, 1))
            cos = torch.cos
            sin = torch.sin

        x_dot[0, 0] = x[3, 0] * cos(x[2, 0]) + u2[0, 0]
        x_dot[1, 0] = x[3, 0] * sin(x[2, 0]) + u2[1, 0]
        x_dot[2, 0] = u1[0, 0]
        x_dot[3, 0] = u1[1, 0]
        return x_dot
