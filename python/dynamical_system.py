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
# Base class for all continuous-time dynamical systems. Supports numerical
# integration and linearization.
#
################################################################################

import torch
import numpy as np

class DynamicalSystem(object):
    """ Base class for all dynamical systems. """

    def __init__(self, x_dim, u_dim):
        """
        Initialize with number of state/control dimensions.

        :param x_dim: number of state dimensions
        :type x_dim: uint
        :param u_dim: number of control dimensions
        :type u_dim: uint
        """
        self._x_dim = x_dim
        self._u_dim = u_dim

    def __call__(self, x, u):
        """
        Compute the time derivative of state for a particular state/control.

        :param x: current state
        :type x: torch.Tensor or np.array
        :param u: current control input
        :type u: torch.Tensor or np.array
        :return: current time derivative of state
        :rtype: torch.Tensor or np.array
        """
        raise NotImplementedError("__call__() has not been implemented.")

    def integrate(self, x0, u, T, dt):
        """
        Integrate initial state x0 (applying constant control u)
        over a time interval of length T, using a time discretization
        of dt.

        :param x0: initial state
        :type x0: np.array
        :param u: control input
        :type u: np.array
        :param T: time interval
        :type T: float
        :param dt: time discretization
        :type dt: float
        :return: state after time T
        :rtype: np.array
        """
        t = 0.0
        x = x0.copy()
        while t < T - 1e-8:
            # Make sure we don't step past T.
            step = min(dt, T - t)

            # Use Runge-Kutta order 4 integration. For details please refer to
            # https://en.wikipedia.org/wiki/Runge-Kutta_methods.
            k1 = step * self.__call__(x, u)
            k2 = step * self.__call__(x + 0.5 * k1, u)
            k3 = step * self.__call__(x + 0.5 * k2, u)
            k4 = step * self.__call__(x + k3, u)

            x += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            t += step

        return x

    def linearize(self, x0, u0):
        """
        Compute the Jacobian linearization of the dynamics for a particular
        state `x0` and control `u0`. Outputs `A` and `B` matrices of a linear
        system:
                   ```\dot x - f(x0, u0) = A (x - x0) + B (u - u0)```

        :param x: state
        :type x: np.array
        :param u: control input
        :type u: np.array
        :return: (A, B) matrices of linearized system
        :rtype: np.array, np.array
        """
        x_torch = torch.from_numpy(x0).requires_grad_(True)
        u_torch = torch.from_numpy(u0).requires_grad_(True)

        x_dot = self.__call__(x_torch, u_torch)

        x_gradient_list = []
        u_gradient_list = []
        for ii in range(self._x_dim):
            x_gradient_list.append(torch.autograd.grad(
                x_dot[ii, 0], x_torch, retain_graph=True)[0])
            u_gradient_list.append(torch.autograd.grad(
                x_dot[ii, 0], u_torch, retain_graph=True)[0])

        A = torch.cat(x_gradient_list, dim=1).detach().numpy().copy().T
        B = torch.cat(u_gradient_list, dim=1).detach().numpy().copy().T
        return A, B
