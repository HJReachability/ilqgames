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
# Unit tests for different Costs.
#
################################################################################

import numpy as np
import torch
import unittest

from cost import Cost
from semiquadratic_cost import SemiquadraticCost
from player_cost import PlayerCost

SMALL_NUMBER = 1e-4

class TestCost(unittest.TestCase):
    """ Tests for Cost and PlayerCost. """

    def testQuadraticize(self):
        """ Tests that PlayerCost can quadraticize correctly. """
        # TODO: use a more complicated custom test Cost that has cross
        # terms in Hessians.
        x = np.array([[1.0], [1.0]])
        u1 = np.array([[1.0], [1.0]])
        u2 = np.array([[1.0], [1.0]])

        semi0 = SemiquadraticCost(0, 0.0, True)
        semi1 = SemiquadraticCost(1, 0.0, True)
        cost = PlayerCost()
        cost.add_cost(semi0, "x", 1.0)
        cost.add_cost(semi1, "x", 2.0)
        cost.add_cost(semi0, 0, 1.0)
        cost.add_cost(semi1, 0, 2.0)
        cost.add_cost(semi0, 1, 1.0)
        cost.add_cost(semi1, 1, 2.0)

        # Compute what the cost should be.
        expected_cost = max(x[0, 0], 0.0)**2 + 2.0 * max(x[1, 0], 0.0)**2 + \
                        max(u1[0, 0], 0.0)**2 + 2.0 * max(u1[1, 0], 0.0)**2 + \
                        max(u2[0, 0], 0.0)**2 + 2.0 * max(u2[1, 0], 0.0)**2

        # Compute expected gradient and Hessians.
        expected_grad_x = np.array([[2.0], [4.0]])
        expected_hess_x = np.array([[2.0, 0.0], [0.0, 4.0]])
        expected_hess_u1 = expected_hess_x
        expected_hess_u2 = expected_hess_x

        # Quadraticize and compare.
        cost, grad_x, hess_x, [hess_u1, hess_u2] = cost.quadraticize(x, [u1, u2])
        self.assertAlmostEqual(cost, expected_cost, delta=SMALL_NUMBER)
        self.assertLess(np.linalg.norm(grad_x - expected_grad_x), SMALL_NUMBER)
        self.assertLess(np.linalg.norm(hess_x - expected_hess_x), SMALL_NUMBER)
        self.assertLess(np.linalg.norm(hess_u1 - expected_hess_u1), SMALL_NUMBER)
        self.assertLess(np.linalg.norm(hess_u2 - expected_hess_u2), SMALL_NUMBER)

if __name__ == '__main__':
    unittest.main()
