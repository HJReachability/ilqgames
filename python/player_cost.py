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
           Chia-Yin Shih        ( cshih@berkeley.edu )
"""
################################################################################
#
# Container to hold a bunch of different Costs and keep track of the arguments
# to each one. Supports automatic quadraticization.
#
################################################################################

import numpy as np
import torch

from cost import Cost

class PlayerCost(object):
    def __init__(self):
        self._costs = []
        self._args = []
        self._weights = []

    def __call__(self, x, u1, u2):
        """
        Evaluate the game cost function at the current state and controls.
        NOTE: `x`, `u1`, and `u2` are all column vectors.

        :param x: state of the system
        :type x: np.array or torch.Tensor
        :param u1: control for player 1
        :type u1: np.array or torch.Tensor
        :param u2: control for player 2
        :type u2: np.array or torch.Tensor
        :return: scalar value of cost
        :rtype: float or torch.Tensor
        """
        if isinstance(x, np.ndarray):
            assert isinstance(u1, np.ndarray) and isinstance(u2, np.ndarray)
        else:
            assert isinstance(x, torch.Tensor) and \
                isinstance(u1, torch.Tensor) and isinstance(u2, torch.Tensor)

        first_time_through = True
        for cost, arg, weight in zip(self._costs, self._args, self._weights):
            if arg == "x":
                current_term = weight * cost(x)
            elif arg == "u1":
                current_term = weight * cost(u1)
            elif arg == "u2":
                current_term = weight * cost(u2)
            else:
                raise RuntimeError("Unrecognized arg name: " + arg)

            if first_time_through:
                total_cost = current_term
            else:
                total_cost += current_term

            first_time_through = False

        return total_cost

    def add_cost(self, cost, arg, weight=1.0):
        """
        Add a new cost to the game, and specify its argument to be either
        "x", "u1", or "u2". Also assign a weight.

        :param cost: cost function to add
        :type cost: Cost
        :param arg: argument of cost, either "x", "u1", or "u2"
        :type arg: string
        :param weight: multiplicative weight for this cost
        :type weight: float
        """
        assert isinstance(cost, Cost)
        assert arg == "x" or arg == "u1" or arg == "u2"

        self._costs.append(cost)
        self._args.append(arg)
        self._weights.append(weight)

    def quadraticize(self, x, u1, u2):
        """
        Compute a quadratic approximation to the overall cost for a
        particular choice of state `x`, and controls `u1` (player 1) and
        `u2` (player 2).

        Returns the gradient and Hessian of the overall cost such that:
        ```
           cost(x + dx, u1 + du1, u2 + du2) \approx
                cost(x, u1, u2) +
                grad_x^T dx +
                0.5 * (dx^T hess_x dx + du1^T hess_u1 du1 + du2^T hess_u2 du2)
        ```

        NOTE: in the notation of `solve_lq_game.py`:
          * `grad_x = l`
          * `hess_x = Q`
          * `hess_u1 = R1` (`R11` if this cost is for player 1, `R21` else)
          * `hess_u2 = R2` (`R12` if this cost is for player 1, `R22` else)

        :param x: state
        :type x: np.array
        :param u1: control input of player 1
        :type u1: np.array
        :param u2: control input of player 2
        :type u2: np.array
        :return: cost(x, u1, u2), grad_x, hess_x, hess_u1, hess_u2
        :rtype: float, np.array, np.array, np.array, np.array
        """
        # Congert to torch.Tensor format.
        x_torch = torch.from_numpy(x).requires_grad_(True)
        u1_torch = torch.from_numpy(u1).requires_grad_(True)
        u2_torch = torch.from_numpy(u2).requires_grad_(True)

        # Evaluate cost here.
        cost_torch = self.__call__(x_torch, u1_torch, u2_torch)
        cost = cost_torch.item()

        # Compute gradients (and store numpy versions).
        grad_x_torch = torch.autograd.grad(
            cost_torch, x_torch, create_graph=True, allow_unused=True)[0]
        grad_u1_torch = torch.autograd.grad(
            cost_torch, u1_torch, create_graph=True, allow_unused=True)[0]
        grad_u2_torch = torch.autograd.grad(
            cost_torch, u2_torch, create_graph=True, allow_unused=True)[0]

        # Compute Hessians (and store numpy versions), and be careful to
        # catch Nones (which indicate cost not depending on a particular
        # variable).
        hess_x = np.zeros((len(x), len(x)))
        grad_x = np.zeros((len(x), 1))
        if grad_x_torch is not None:
            grad_x = grad_x_torch.detach().numpy().copy()
            for ii in range(len(x)):
                hess_row = torch.autograd.grad(
                    grad_x_torch[ii, 0], x_torch, retain_graph=True)[0]
                hess_x[ii, :] = hess_row.detach().numpy().copy().T

        hess_u1 = np.zeros((len(u1), len(u1)))
        if grad_u1_torch is not None:
            grad_u1 = grad_u1_torch.detach().numpy().copy()
            for ii in range(len(u1)):
                hess_row = torch.autograd.grad(
                    grad_u1_torch[ii, 0], u1_torch, retain_graph=True)[0]
                hess_u1[ii, :] = hess_row.detach().numpy().copy().T

        hess_u2 = np.zeros((len(u2), len(u2)))
        if grad_u2_torch is not None:
            grad_u2 = grad_u2_torch.detach().numpy().copy()
            for ii in range(len(u2)):
                hess_row = torch.autograd.grad(
                    grad_u2_torch[ii, 0], u2_torch, retain_graph=True)[0]
                hess_u2[ii, :] = hess_row.detach().numpy().copy().T

        return cost, grad_x, hess_x, hess_u1, hess_u2
