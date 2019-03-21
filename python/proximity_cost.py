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
import numpy as np
import matplotlib.pyplot as plt

from cost import Cost
from point import Point

class ProximityCost(Cost):
    def __init__(self, position_indices, point,
                 max_distance, outside_weight=0.1, apply_after_time=-1,
                 name=""):
        """
        Initialize with dimension to add cost to and threshold BELOW which
        to impose quadratic cost. Above the threshold, we use a very light
        quadratic cost. The overall cost is continuous.

        :param position_indices: indices of input corresponding to (x, y)
        :type position_indices: (uint, uint)
        :param point: point from which to compute proximity
        :type point: Point
        :param max_distance: maximum value of distance to penalize
        :type threshold: float
        :param outside_weight: weight of quadratic cost outside threshold
        :type outside_weight: float
        :param apply_after_time: only apply proximity time after this time step
        :type apply_after_time: int
        """
        self._x_index, self._y_index = position_indices
        self._point = point
        self._max_squared_distance = max_distance**2
        self._outside_weight = outside_weight
        self._apply_after_time = apply_after_time
        super(ProximityCost, self).__init__(name)

    def __call__(self, x, k=0):
        """
        Evaluate this cost function on the given input state and time.
        NOTE: `x` should be a column vector.

        :param x: concatenated state of the two systems
        :type x: torch.Tensor
        :param k: time step, if cost is time-varying
        :type k: uint
        :return: scalar value of cost
        :rtype: torch.Tensor
        """
        if k < self._apply_after_time:
            return torch.zeros(
                1, 1, requires_grad=True).double()

        # Compute relative distance.
        dx = x[self._x_index, 0] - self._point.x
        dy = x[self._y_index, 0] - self._point.y
        relative_squared_distance = dx*dx + dy*dy

        if relative_squared_distance < self._max_squared_distance:
            return -relative_squared_distance * torch.ones(
                1, 1, requires_grad=True).double()

        # Outside penalty is:
        #   ``` outside_weight * (relative_distance - max_distance)**2 ```
        # which can be computed from what we have already with only one sqrt.
        outside_penalty = self._outside_weight * (
            relative_squared_distance + self._max_squared_distance -
            2.0 * torch.sqrt(
                relative_squared_distance * self._max_squared_distance))
        return -outside_penalty - self._max_squared_distance * torch.ones(
            1, 1, requires_grad=True).double()

    def render(self, ax=None):
        """ Render this obstacle on the given axes. """
        if np.isinf(self._max_squared_distance):
            radius = 1.0
        else:
            radius = np.sqrt(self._max_squared_distance)

        circle = plt.Circle(
            (self._point.x, self._point.y), radius,
            color="g", fill=True, alpha=0.75)
        ax.add_artist(circle)
        ax.text(self._point.x + 1.25, self._point.y + 1.25, "goal", fontsize=10)
