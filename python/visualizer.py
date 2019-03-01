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
# Fancy visualization class.
#
################################################################################

import numpy as np
import matplotlib.pyplot as plt

class Visualizer(object):
    def __init__(self, x_idx, y_idx, obs_centers, obs_radii, goal_center, figure_number=1):
        """
        Construct from indices of x/y coordinates in state vector.

        :param x_idx: index of x-coordinate of state
        :type x_idx: uint
        :param y_idx: index of y-coordinate of state
        :type y_idx: uint
        :param obs_centers: list of obstacle center points
        :type obs_centers: [Point]
        :param obs_radii: list of obstacle radii
        :type obs_radii: [float]
        :param goal_center: position of the goal
        :type goal_center: Point
        :param figure_number: which figure number to operate on
        :type figure_number: uint
        """
        self._x_idx = x_idx
        self._y_idx = y_idx
        self._figure_number = figure_number
        self._obs_centers = obs_centers
        self._obs_radii = obs_radii
        self._goal_center = goal_center

        # Store history as list of trajectories.
        # Each trajectory is a dictionary of lists of states and controls.
        self._iterations = []
        self._history = []

    def add_trajectory(self, iteration, traj):
        """
        Add a new trajectory to the history.

        :param iteration: which iteration is this
        :type iteration: uint
        :param traj: trajectory
        :type traj: {"xs": [np.array], "u1s": [np.array], "u2s": [np.array]}
        """
        self._iterations.append(iteration)
        self._history.append(traj)

    def plot(self):
        """ Plot everything. """
        plt.figure(self._figure_number)

        # Plot the obstacles.
        ax = plt.gca()
        for center, radius in zip(self._obs_centers, self._obs_radii):
            circle = plt.Circle(
                (center.x, center.y), radius, color='r', fill=False)
            ax.add_artist(circle)

        # Plot the goal.
        circle = plt.Circle(
            (self._goal_center.x, self._goal_center.y),
            0.5, color='b', fill=True)
        ax.add_artist(circle)

        # Plot the history of trajectories.
        for ii, traj in zip(self._iterations, self._history):
            xs = [x[self._x_idx, 0] for x in traj["xs"]]
            ys = [x[self._y_idx, 0] for x in traj["xs"]]
            plt.plot(xs, ys, label="Iteration " + str(ii))

        plt.legend()

#        plt.figure(self._figure_number + 1)
#        for ii, traj in zip(self._iterations, self._history):
#            plt.plot([np.linalg.norm(u1) for u1 in traj["u1s"]], label=str(ii) + ": u1")

#        plt.legend()

#        plt.figure(self._figure_number + 2)
#        for ii, traj in zip(self._iterations, self._history):
#            plt.plot([np.linalg.norm(u2) for u2 in traj["u2s"]], label=str(ii) + ": u2")

#        plt.legend()
