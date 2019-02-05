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

Author(s): Chia-Yin Shih (cshih@berkeley.edu)
"""

import torch
import numpy as np

def quadratize(c, x0, u0):
    """
    Compute the quadratic approximation of the cost objective
    for a given state `x0` and `u0.` Outputs `Q` and `f` of
    the following equation:

        c(x,u) = c(x0,u0) + f^Tz + 1/2z^T Qz

    where
        z = [x-x0; u-u0]

    :param c: cost function that takes in state and input
    :type c: a python function
    :param x: state
    :type x: np.array
    :param u: control input
    :type u: np.array
    :return: (Q, f) matrices of quadratized cost
    :rtype: np.array, np.array
    """
    x_torch = torch.from_numpy(x0).requires_grad_(True)
    u_torch = torch.from_numpy(u0).requires_grad_(True)

    c_x0_u0 = c(x_torch, u_torch)

    x_dim = len(x0)
    u_dim = len(u0)
    xu_dim = x_dim + u_dim

    # Compute f.
    x_deriv_torch = torch.autograd.grad(c_x0_u0, x_torch, create_graph=True)

    u_deriv_torch = torch.autograd.grad(c_x0_u0, u_torch, create_graph=True)

    x_deriv = x_deriv_torch[0].detach().numpy().copy()
    u_deriv = u_deriv_torch[0].detach().numpy().copy()

    f = np.append(x_deriv, u_deriv)

    # Compute Q.
    Q = np.zeros((xu_dim, xu_dim))

    # Compute dxx.
    for ii in range(x_dim):
        curr_x_deriv = torch.autograd.grad(
            x_deriv_torch[0][ii], x_torch, create_graph=True)
        Q[ii, :x_dim] = np.reshape(
            curr_x_deriv[0].detach().numpy().copy(), (x_dim,))

    # Compute duu.
    for ii in range(u_dim):
        curr_u_deriv = torch.autograd.grad(
            u_deriv_torch[0][ii], u_torch, create_graph=True)
        Q[x_dim + ii, x_dim:] = np.reshape(
            curr_u_deriv[0].detach().numpy().copy(), (u_dim,))


    # Compute dxdu.
    for ii in range(x_dim):
        curr_u_deriv = torch.autograd.grad(
            x_deriv_torch[0][ii], u_torch, create_graph=True)
        Q[ii,x_dim:] = np.reshape(
            curr_u_deriv[0].detach().numpy().copy(), (u_dim,))

    # Compute dudx based on symmetry.
    Q[x_dim:, :x_dim] = Q[:x_dim, x_dim:].T

    return Q, f
