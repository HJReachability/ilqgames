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
# Proximity cost, derived from Cost base class. Implements a cost function that
# depends only on state and penalizes -min(distance, max_distance)^2.
#
################################################################################

import torch

from cost import Cost
from point import Point

class ProximityCost(Cost):
    def __init__(self, position_indices, point, max_distance):
        """
        Initialize with dimension to add cost to and threshold BELOW which
        to impose quadratic cost.

        :param position_indices: indices of input corresponding to (x, y)
        :type position_indices: (uint, uint)
        :param point: point from which to compute proximity
        :type point: Point
        :param max_distance: maximum value of distance to penalize
        :type threshold: float
        """
        self._x_index, self._y_index = position_indices
        self._point = point
        self._max_squared_distance = max_distance**2
        super(ProximityCost, self).__init__()

    def __call__(self, x):
        """
        Evaluate this cost function on the given input state.
        NOTE: `x` should be a column vector.

        :param x: concatenated state of the two systems
        :type x: torch.Tensor
        :return: scalar value of cost
        :rtype: torch.Tensor
        """
        # Compute relative distance.
        dx = x[self._x_index, 0] - self._point.x
        dy = x[self._y_index, 0] - self._point.y
        relative_squared_distance = dx*dx + dy*dy

        if relative_squared_distance < self._max_squared_distance:
            return -relative_squared_distance * torch.ones(1, 1)

        return -self._max_squared_distance * torch.ones(1, 1)

class ConcatenatedStateProximityCost(Cost):
    def __init__(self, position_indices1, position_indices2, max_distance):
        """
        Initialize with dimension to add cost to and threshold BELOW which
        to impose quadratic cost.

        :param position_indices1: indices of input corresponding to (x, y) for
          vehicle 1
        :type position_indices1: (uint, uint)
        :param position_indices2: indices of input corresponding to (x, y) for
          vehicle 2
        :type position_indices2: (uint, uint)
        :param max_distance: maximum value of distance to penalize
        :type threshold: float
        """
        self._x_index1, self._y_index1 = position_indices1
        self._x_index2, self._y_index2 = position_indices2
        self._max_squared_distance = max_distance**2
        super(ProximityCost, self).__init__()

    def __call__(self, xu):
        """
        Evaluate this cost function on the given input, which might either be
        a state `x` or a control `u`. Hence the input is named `xu`.
        NOTE: `xu` should be a PyTorch tensor with `requires_grad` set `True`.
        NOTE: `xu` should be a column vector.

        Here, `xu` is just the concatenated state of the two systems.

        :param xu: concatenated state of the two systems
        :type xu: torch.Tensor
        :return: scalar value of cost
        :rtype: torch.Tensor
        """
        # Compute relative distance.
        dx = xu[self._x_index1, 0] - xu[self._x_index2, 0]
        dy = xu[self._y_index1, 0] - xu[self._y_index2, 0]
        relative_squared_distance = dx*dx + dy*dy

        if relative_squared_distance < self._max_squared_distance:
            return -relative_squared_distance

        return -self._max_squared_distance * torch.ones(1, 1)
