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
# Base class for all two-player continuous-time dynamical systems. Supports 
# numerical integration and linearization.
#
################################################################################

import torch
import numpy as np
import scipy as sp

class TwoPlayerDynamicalSystem(object):
    """ Base class for all two-player dynamical systems. """

    def __init__(self, x_dim, u1_dim, u2_dim, T=0.1):
        """
        Initialize with number of state/control dimensions.

        :param x_dim: number of state dimensions
        :type x_dim: uint
        :param u1_dim: number of control dimensions for player 1
        :type u1_dim: uint
        :param u2_dim: number of control dimensions for player 2
        :type u2_dim: uint
        :param T: time interval
        :type T: float
        """
        self._x_dim = x_dim
        self._u1_dim = u1_dim
        self._u2_dim = u2_dim
        self._T = T

    def __call__(self, x, u1, u2):
        """
        Compute the time derivative of state for a particular state/control.
        NOTE: `x`, `u1`, and `u2` should be 2D (i.e. column vectors).

        :param x: current state
        :type x: torch.Tensor or np.array
        :param u1: current control input for player 1
        :type u1: torch.Tensor or np.array
        :param u2: current control input for player 2
        :type u2: torch.Tensor or np.array
        :return: current time derivative of state
        :rtype: torch.Tensor or np.array
        """
        raise NotImplementedError("__call__() has not been implemented.")

    def integrate(self, x0, u1, u2, dt=None):
        """
        Integrate initial state x0 (applying constant controls u1 and 
        u2) over a time interval of length T, using a time discretization
        of dt.

        :param x0: initial state
        :type x0: np.array
        :param u1: control input for player 1
        :type u1: np.array
        :param u2: control input for player 2
        :type u2: np.array
        :param dt: time discretization
        :type dt: float
        :return: state after time T
        :rtype: np.array
        """
        if dt is None:
            dt = 0.1 * self._T

        t = 0.0
        x = x0.copy()
        while t < self._T - 1e-8:
            # Make sure we don't step past T.
            step = min(dt, self._T - t)

            # Use Runge-Kutta order 4 integration. For details please refer to
            # https://en.wikipedia.org/wiki/Runge-Kutta_methods.
            k1 = step * self.__call__(x, u1, u2)
            k2 = step * self.__call__(x + 0.5 * k1, u1, u2)
            k3 = step * self.__call__(x + 0.5 * k2, u1, u2)
            k4 = step * self.__call__(x + k3, u1, u2)

            x += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            t += step

        return x

    def linearize(self, x0, u10, u20):
        """
        Compute the Jacobian linearization of the dynamics for a particular
        state `x0` and controls `u10` and `u20`. Outputs `A` and `B` matrices 
        of a linear system:
                   ```\dot x - f(x0, u10, u20) = A (x - x0) + B1 (u1 - u10) + B2 (u2 - u20)```

        :param x0: state
        :type x0: np.array
        :param u10: control input for player 1
        :type u10: np.array
        :param u20: control input for player 2
        :type u20: np.array
        :return: (A, B1, B2) matrices of linearized system
        :rtype: np.array, np.array
        """
        x_torch = torch.from_numpy(x0).requires_grad_(True)
        u1_torch = torch.from_numpy(u10).requires_grad_(True)
        u2_torch = torch.from_numpy(u20).requires_grad_(True)

        x_dot = self.__call__(x_torch, u1_torch, u2_torch)

        x_gradient_list = []
        u1_gradient_list = []
        u2_gradient_list = []
        for ii in range(self._x_dim):
            x_gradient_list.append(torch.autograd.grad(
                x_dot[ii, 0], x_torch, retain_graph=True)[0])
            u1_gradient_list.append(torch.autograd.grad(
                x_dot[ii, 0], u1_torch, retain_graph=True)[0])
            u2_gradient_list.append(torch.autograd.grad(
                x_dot[ii, 0], u2_torch, retain_graph=True)[0])

        A = torch.cat(x_gradient_list, dim=1).detach().numpy().copy().T
        B1 = torch.cat(u1_gradient_list, dim=1).detach().numpy().copy().T
        B2 = torch.cat(u2_gradient_list, dim=1).detach().numpy().copy().T
        return A, B1, B2

    def linearize_discrete(self, x0, u10, u20):
        """
        Compute the Jacobian linearization of the dynamics for a particular
        state `x0` and controls `u10` and `u20`. Outputs `A`, `B1`, and 
        `B2` matrices and `c` offset vector of a discrete-time linear system:
                   ```x(k + 1) - x0 = A (x(k) - x0) + B1 (u1(k) - u10) + B2 (u2(k) - u20) + c```

        :param x0: state
        :type x0: np.array
        :param u10: control input player 1
        :type u10: np.array
        :param u20: control input player 2
        :type u20: np.array
        :return: (A, B1, B2, c) matrices and offset vector of the discrete-time 
                 linearized system
        :rtype: np.array, np.array, np.array, np.array
        """
        A_cont, B1_cont, B2_cont = self.linearize(x0, u10, u20)
        c_cont = self.__call__(x0, u10, u20)

        eAT = sp.linalg.expm(A * self._T)
        Ainv = np.linalg.inv(A)

        # See https://en.wikipedia.org/wiki/Discretization#Discretization_of_linear_state_space_models
        # for derivation of discrete-time from continuous time linear system. 
        A_disc = eAT
        B1_disc = Ainv @ (eAT - np.eye(self._x_dim)) @ B1_cont
        B2_disc = Ainv @ (eAT - np.eye(self._x_dim)) @ B2_cont
        c_disc = Ainv @ (eAT - np.eye(self._x_dim)) @ c_cont

        return A_disc, B1_disc, B2_disc, c_disc

