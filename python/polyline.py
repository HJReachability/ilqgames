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
# Polyline class to represent piecewise linear path in 2D.
#
################################################################################

from point import Point
from line_segment import LineSegment

class Polyline(object):
    """ Class to represent piecewise linear path in 2D. """

    def __init__(self, points=[]):
        """
        Initialize from a list of points. Keeps only a reference to input list.

        :param points: list of Points
        :type points: [Point]
        """
        self.points = points

    def signed_distance_to(self, point):
        """
        Compute signed distance from this polyline to the given point.
        Sign convention is positive to the right and negative to the left, e.g.:
                                        *
                                        |
                   negative             |             positive
                                        |
                                        |
                                        *

        :param point: query point
        :type point: Point
        """
        # NOTE: for now, we'll just implement this with a naive linear search.
        # In future, if we want to optimize at all we can pass in a guess of
        # which index we expect the closest point to be close to.
        best_signed_distance = float("inf")
        for ii in range(1, len(self.points)):
            segment = LineSegment(self.points[ii - 1], self.points[ii])
            signed_distance = segment.signed_distance_to(point)

            if abs(signed_distance) < abs(best_signed_distance):
                best_signed_distance = signed_distance

        return best_signed_distance
