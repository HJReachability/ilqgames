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
# Unit tests for Unicycle4D.
#
################################################################################

import numpy as np
import torch
import unittest

from unicycle_4d import Unicycle4D

SMALL_NUMBER = 1e-2

def wrap_angle(angle):
    return -np.pi + (angle + np.pi) % (2.0 * np.pi)

class TestUnicycle4D(unittest.TestCase):
    """ Tests for Unicycle4D. """

    def testIntegrate(self):
        """
        Tests that if we maintain a constant speed and turning rate
        we go in a circle.
        """
        # Set fixed control inputs omega (rad/s) and acceleration (m/s/s).
        OMEGA = 0.5
        ACCELERATION = 0.0
        U = np.array([[OMEGA], [ACCELERATION]])

        # Initial state.
        SPEED = 8.0
        X0 = np.array([[0.0], [0.0], [0.0], [SPEED]])

        # Compute turning radius (m) and time per full circle (s).
        TURNING_RADIUS = SPEED / OMEGA
        TIME_PER_CIRCLE = 2.0 * np.pi / OMEGA

        # Check that we end up where we started if we end up after
        # TIME_PER_CYCLE elapses.
        dynamics = Unicycle4D(T=TIME_PER_CIRCLE)
        x_final = dynamics.integrate(X0, U)
        x_final[2, 0] = wrap_angle(x_final[2, 0])
        self.assertLess(np.linalg.norm(X0 - x_final), SMALL_NUMBER)

        # Check that after half a circle we're in the right place.
        dynamics = Unicycle4D(T=0.5 * TIME_PER_CIRCLE)
        x_final = dynamics.integrate(X0, U)
        self.assertAlmostEqual(x_final[0, 0], X0[0, 0], delta=SMALL_NUMBER)
        self.assertAlmostEqual(x_final[1, 0], X0[1, 0] + 2.0 * TURNING_RADIUS,
                               delta=SMALL_NUMBER)
        self.assertAlmostEqual(wrap_angle(x_final[2, 0]),
                               wrap_angle(np.pi - X0[2, 0]),
                               delta=SMALL_NUMBER)
        self.assertAlmostEqual(x_final[3, 0], X0[3, 0], delta=SMALL_NUMBER)

    def testJacobian(self):
        dynamics = Unicycle4D()

        # Custom Jacobian.
        def jacobian(x0, u0):
            A = np.array([
                [0.0, 0.0, -x0[3, 0] * np.sin(x0[2, 0]), np.cos(x0[2, 0])],
                [0.0, 0.0, x0[3, 0] * np.cos(x0[2, 0]), np.sin(x0[2, 0])],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0]])
            B = np.array([[0.0, 0.0],
                          [0.0, 0.0],
                          [1.0, 0.0],
                          [0.0, 1.0]])
            return A, B

        # Pick a bunch of random states and controls and compare Jacobians.
        NUM_RANDOM_TRIALS = 100
        for ii in range(NUM_RANDOM_TRIALS):
            x0 = np.random.normal(0.0, 1.0, (dynamics._x_dim, 1))
            u0 = np.random.normal(0.0, 1.0, (dynamics._u_dim, 1))

            test_A, test_B = dynamics.linearize(x0, u0)
            custom_A, custom_B = jacobian(x0, u0)

            self.assertLess(np.linalg.norm(test_A - custom_A), SMALL_NUMBER)
            self.assertLess(np.linalg.norm(test_B - custom_B), SMALL_NUMBER)

    def testLinearizeDiscrete(self):
        dynamics = Unicycle4D(T=0.1)

        # Pick a bunch of random states and controls and compare discrete-time
        # linearization with an Euler approximation.
        NUM_RANDOM_TRIALS = 100
        for ii in range(NUM_RANDOM_TRIALS):
            x0 = np.random.normal(0.0, 1.0, (dynamics._x_dim, 1))
            u0 = np.random.normal(0.0, 1.0, (dynamics._u_dim, 1))

            test_A_cont, test_B_cont = dynamics.linearize(x0, u0)
            test_c_cont = dynamics(x0, u0)

            test_A_disc, test_B_disc, test_c_disc = dynamics.linearize_discrete(x0, u0)

            # The Euler approximation to \dot{x} = Ax + Bu + c gives
            # x(k + 1) = x(k) + ATx(k) + BTu(k) + cT
            # So we want A_disc to be close to (I + AT), B_disc to be close to
            # BT, and c_disc to be close to c.
            LESS_SMALL_NUMBER = 5.0 * dynamics._T
            self.assertLess(np.linalg.norm(test_A_disc -
                                           (np.eye(dynamics._x_dim) + test_A_cont * dynamics._T)),
                            LESS_SMALL_NUMBER)
            self.assertLess(np.linalg.norm(test_B_disc - test_B_cont * dynamics._T),
                            LESS_SMALL_NUMBER)
            self.assertLess(np.linalg.norm(test_c_disc - test_c_cont * dynamics._T),
                            LESS_SMALL_NUMBER)


if __name__ == '__main__':
    unittest.main()
