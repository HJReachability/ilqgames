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
# Obstacle cost, derived from Cost base class. Implements a cost function that
# depends only on state and penalizes min(0, dist - max_distance)^2.
#
################################################################################

import torch
import numpy as np
import matplotlib.pyplot as plt

from cost import Cost
from point import Point

class ObstacleCost(Cost):
    def __init__(self, position_indices, point, max_distance, name=""):
        """
        Initialize with dimension to add cost to and a max distance beyond
        which we impose no additional cost.

        :param position_indices: indices of input corresponding to (x, y)
        :type position_indices: (uint, uint)
        :param point: center of the obstacle from which to compute distance
        :type point: Point
        :param max_distance: maximum value of distance to penalize
        :type threshold: float
        """
        self._x_index, self._y_index = position_indices
        self._point = point
        self._max_distance = max_distance
        super(ObstacleCost, self).__init__(name)

    def __call__(self, x, k=0):
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
        relative_distance = torch.sqrt(dx*dx + dy*dy)

        return min(relative_distance - self._max_distance, torch.zeros(
            1, 1, requires_grad=True).double())**2

    def render(self, ax=None):
        """ Render this obstacle on the given axes. """
        circle = plt.Circle(
            (self._point.x, self._point.y), self._max_distance,
            color="r", fill=True, alpha=0.75)
        ax.add_artist(circle)
        ax.text(self._point.x - 1.25, self._point.y - 1.25, "obs", fontsize=8)
