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

Author(s): Chia-Yin Shih ( cshih@berkeley.edu )
"""
################################################################################
#
# Unit tests for quadratize_cost.
#
################################################################################

import numpy as np
import torch
import unittest
from quadratize_cost import quadratize

SMALL_NUMBER = 1e-2

class TestQuadratizeCost(unittest.TestCase):
    """ Tests for quadratize. """

    def testQuadratize(self):

        def cost(x, u):
            """
            c(x,u) = 2x1^2 + x2^2u2u1 + u1^2

            """
            return 2 * pow(x[0],2) + pow(x[1],2)*u[0]*u[1] + pow(u[1],2)

        def quadratize_ground_truth(x, u):
            """

            f = [4x1, 2u1u2x2, x2^2u2 + 2u1, x2^2u1]

            Q = 
                4  0        0      0 
                0  2u1u2  2x2u2   2u1x2
                0  2u2x2    2      x2^2
                0  2u1x2   x2^2    0
            """
            f = np.zeros(4)
            f[0] = 2 * x[0]
            f[1] = 2 * u[0] * u[1] * x[1]
            f[2] = x[1]*x[1]*u[1] + 2*u[0]
            f[3] = x[1]*x[1]*u[0]

            Q = np.zeros((4,4))
            Q[0,0] = 4
            Q[1,1] = 2*u[0]*u[1]
            Q[2,2] = 2
            Q[1,2] = 2*x[1]*u[1]
            Q[1,3] = 2*x[1]*u[0]
            Q[2,3] = x[1]*x[1]
            Q[2,1] = Q[1,2]
            Q[3,1] = Q[1,3]
            Q[3,2] = Q[2,3]

            return f, q

        # Pick a bunch of random states and controls and compare quadratize Q and f
        NUM_RANDOM_TRIALS = 100
        X_DIM = 2
        U_DIM = 2
        for ii in range(NUM_RANDOM_TRIALS):
            x0 = np.random.normal(0.0, 1.0, (X_DIM, 1))
            u0 = np.random.normal(0.0, 1.0, (U_DIM, 1))

            test_f, test_Q = quadratize(x0, u0)
            true_f, true_Q = quadratize_ground_truth(x0, u0)

            self.assertLess(np.linalg.norm(test_f - true_f), SMALL_NUMBER)
            self.assertLess(np.linalg.norm(test_Q - true_Q), SMALL_NUMBER)

if __name__ == '__main__':
    unittest.main()
