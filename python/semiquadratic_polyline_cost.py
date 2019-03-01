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

from cost import Cost
from point import Point
from polyline import Polyline

class SemiquadraticPolylineCost(Cost):
    def __init__(self, polyline, signed_distance_threshold, oriented_right,
                 position_indices, name):
        """
        Initialize with a polyline, a threshold in signed distance from the
        polyline, and an orientation. If `oriented_right` is `True`, that
        indicates that the cost starts taking effect once signed distance
        exceeds the signed distance threshold. If `False`, that indicates
        that the cost will take effect once signed distance falls below the
        threshold.

        :param polyline: piecewise linear path which defines signed distances
        :type polyline: Polyline
        :param signed_distance_threshold: value above/below which to penalize
        :type signed_distance_threshold: float
        :param oriented_right: Boolean flag determining which side of threshold
          to penalize
        :type oriented_right: bool
        :param position_indices: indices of input corresponding to (x, y)
        :type position_indices: (uint, uint)
        """
        self._polyline = polyline
        self._signed_distance_threshold = signed_distance_threshold
        self._oriented_right = oriented_right
        self._x_index, self._y_index = position_indices
        super(SemiquadraticPolylineCost, self).__init__(name)

    def __call__(self, xu):
        """
        Evaluate this cost function on the given input, which might either be
        a state `x` or a control `u`. Hence the input is named `xu`.
        NOTE: `xu` should be a PyTorch tensor with `requires_grad` set `True`.
        NOTE: `xu` should be a column vector.

        :param xu: state of the system
        :type xu: torch.Tensor
        :return: scalar value of cost
        :rtype: torch.Tensor
        """
        signed_distance = self._polyline.signed_distance_to(
            Point(xu[self._x_index, 0], xu[self._y_index, 0]))

        if self._oriented_right:
            if signed_distance > self._signed_distance_threshold:
                return (signed_distance - self._signed_distance_threshold) ** 2
        else:
            if signed_distance < self._signed_distance_threshold:
                return (self._signed_distance_threshold - signed_distance) ** 2

        return torch.zeros(1, 1)
