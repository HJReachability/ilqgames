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
# Class for 2D line segments.
#
################################################################################

class LineSegment(object):
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def __len__(self):
        return (self.p1 - self.p2).norm()

    def signed_distance_to(self, point):
        """
        Compute signed distance to other point.
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
        # Vector from p1 to query.
        relative = point - self.p1

        # Compute the unit direction of this line segment.
        direction = self.p2 - self.p1
        direction /= direction.norm()

        # Find signed length of projection and of cross product.
        projection = relative.x * direction.x + relative.y * direction.y
        cross = relative.x * direction.y - direction.x * relative.y
        cross_sign = 1.0 if cross >= 0.0 else -1.0

        if projection < 0.0:
            # Query lies behind this line segment, so closest distance will be
            # from p1.
            return cross_sign * relative.norm()
        elif projection > self.__len__():
            # Closest distance will be to p2.
            return cross_sign * (self.p2 - point).norm()
        else:
            return cross
