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
# Semiquadratic cost that takes effect a fixed distance away from a Polyline.
#
################################################################################

import torch
import matplotlib.pyplot as plt

from cost import Cost
from point import Point
from polyline import Polyline

class SemiquadraticPolylineCost(Cost):
    def __init__(self, polyline, distance_threshold, position_indices, name=""):
        """
        Initialize with a polyline, a threshold in distance from the polyline.

        :param polyline: piecewise linear path which defines signed distances
        :type polyline: Polyline
        :param distance_threshold: value above which to penalize
        :type distance_threshold: float
        :param position_indices: indices of input corresponding to (x, y)
        :type position_indices: (uint, uint)
        """
        self._polyline = polyline
        self._distance_threshold = distance_threshold
        self._x_index, self._y_index = position_indices
        super(SemiquadraticPolylineCost, self).__init__(name)

    def __call__(self, x, k=0):
        """
        Evaluate this cost function on the given state and time.
        NOTE: `x` should be a PyTorch tensor with `requires_grad` set `True`.
        NOTE: `x` should be a column vector.

        :param x: state of the system
        :type x: torch.Tensor
        :return: scalar value of cost
        :rtype: torch.Tensor
        """
        signed_distance = self._polyline.signed_distance_to(
            Point(x[self._x_index, 0], x[self._y_index, 0]))

        if abs(signed_distance) > self._distance_threshold:
            return (abs(signed_distance) - self._distance_threshold) ** 2

        return torch.zeros(1, 1, requires_grad=True).double()

    def render(self, ax=None):
        """ Render this cost on the given axes. """
        xs = [pt.x for pt in self._polyline.points]
        ys = [pt.y for pt in self._polyline.points]
        ax.plot(xs, ys, "k", alpha=0.25)
