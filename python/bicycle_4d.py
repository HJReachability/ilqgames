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
# 4D (kinematic) bicycle model. Dynamics are as follows:
#                          \dot x     = v cos(psi + beta)
#                          \dot y     = v sin(psi + beta)
#                          \dot psi   = (v / l_r) sin(beta)
#                          \dot v     = u1
#                 where beta = arctan((l_r / (l_f + l_r)) tan(u2))
#
# Dynamics were taken from:
# https://borrelli.me.berkeley.edu/pdfpub/IV_KinematicMPC_jason.pdf
#
# `psi` is the inertial heading.
# `beta` is the angle of the current velocity of the center of mass with respect
#     to the longitudinal axis of the car
# `u1` is the acceleration of the center of mass in the same direction as the
#     velocity.
# `u2` is the front steering angle.
#
################################################################################

import torch
import numpy as np

from dynamical_system import DynamicalSystem

class Bicycle4D(DynamicalSystem):
    """ 4D unicycle model. """

    def __init__(self, l_f, l_r, T=0.1):
        """
        Initialize with front and rear lengths.

        :param l_f: distance (m) between center of mass and front axle
        :type l_f: float
        :param l_r: distance (m) between center of mass and rear axle
        :type l_r: float
        """
        self._l_f = l_f
        self._l_r = l_r
        super(Bicycle4D, self).__init__(4, 2, T)

    def __call__(self, x, u):
        """
        Compute the time derivative of state for a particular state/control.
        NOTE: `x` and `u` should be 2D (i.e. column vectors).

        :param x: current state
        :type x: torch.Tensor or np.array
        :param u: current control input
        :type u: torch.Tensor or np.array
        :return: current time derivative of state
        :rtype: torch.Tensor or np.array
        """
        if isinstance(x, np.ndarray):
            assert isinstance(u, np.ndarray)
            x_dot = np.zeros((self._x_dim, 1))
            cos = np.cos
            sin = np.sin
            tan = np.tan
            atan = np.arctan
        else:
            assert isinstance(u, torch.Tensor)
            x_dot = torch.zeros((self._x_dim, 1))
            cos = torch.cos
            sin = torch.sin
            tan = torch.tan
            atan = torch.atan

        beta = atan((self._l_r / (self._l_f + self._l_r)) * tan(u[1, 0]))

        x_dot[0, 0] = x[3, 0] * cos(x[2, 0] + beta)
        x_dot[1, 0] = x[3, 0] * sin(x[2, 0] + beta)
        x_dot[2, 0] = (x[3, 0] / self._l_r) * sin(beta)
        x_dot[3, 0] = u[0, 0]
        return x_dot
