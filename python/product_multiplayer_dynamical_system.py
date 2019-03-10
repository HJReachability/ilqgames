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
# Implements a multiplayer dynamical system who's dynamics decompose into a
# Cartesian product of single-player dynamical systems.
#
################################################################################

import torch
import numpy as np
import scipy as sp
from scipy.linalg import block_diag

from multiplayer_dynamical_system import MultiPlayerDynamicalSystem

class ProductMultiPlayerDynamicalSystem(MultiPlayerDynamicalSystem):
    def __init__(self, subsystems, T=0.1):
        """
        Initialize with a list of dynamical systems.

        :param subsystems: list of component (single-player) dynamical systems
        :type subsystems: [DynamicalSystem]
        :param T: time interval
        :type T: float
        """
        self._subsystems = subsystems
        self._x_dims = [sys._x_dim for sys in subsystems]

        x_dim = sum(self._x_dims)
        u_dims = [sys._u_dim for sys in subsystems]
        super(ProductMultiPlayerDynamicalSystem, self).__init__(
            x_dim, u_dims, T)

    def __call__(self, x, u):
        """
        Compute the time derivative of state for a particular state/control.
        NOTE: `x`, and all `u` should be 2D (i.e. column vectors).

        :param x: current state
        :type x: torch.Tensor or np.array
        :param u: list of current control inputs for all each player
        :type u: [torch.Tensor] or [np.array]
        :return: current time derivative of state
        :rtype: torch.Tensor or np.array
        """
        subsystem_xs = np.split(x, np.cumsum(self._x_dims[:-1]), axis=0)

        x_dot_list = [sys(x0, u0)
                      for sys, x0, u0 in zip(self._subsystems, subsystem_xs, u)]
        return np.concatenate(x_dot_list, axis=0)

    def linearize(self, x0, u0):
        """
        Compute the Jacobian linearization of the dynamics for a particular
        state `x0` and controls `u0` for each player. Outputs `A` and `Bi`
        matrices of a linear system:
          ```\dot x - f(x0, u0) = A (x - x0) + sum_i Bi (ui - ui0) ```

        NOTE: Overrides the implementation in the base class for efficiency,
        since we know the A matrix is block diagonal and the Bi matrices are
        the B's from the individual subsystem Jacobians padded vertically with
        zeros.

        :param x0: state
        :type x0: np.array
        :param u0: list of control inputs for each player
        :type u0: [np.array]
        :return: (A, [Bi]) matrices of linearized system
        :rtype: np.array, [np.array]
        """
        subsystem_xs = np.split(x0, np.cumsum(self._x_dims[:-1]), axis=0)
        subsystem_linearizations = [
            sys.linearize(x, u)
            for sys, x, u in zip(self._subsystems, subsystem_xs, u0)]

        subsystem_As = [AB[0] for AB in subsystem_linearizations]
        subsystem_Bs = [AB[1] for AB in subsystem_linearizations]

        # A is block diagonal.
        A = block_diag(*subsystem_As)

        # Each Bi is [0; ...; 0; Bi; 0; ...; 0].
        def create_B(ii, Bi):
            before_rows = sum(self._x_dims[:ii])
            after_rows = self._x_dim - before_rows - self._x_dims[ii]
            cols = self._u_dims[ii]
            return np.concatenate([np.zeros((before_rows, cols)),
                                   Bi,
                                   np.zeros((after_rows, cols))], axis=0)

        B = [create_B(ii, Bi) for ii, Bi in enumerate(subsystem_Bs)]

        return A, B
