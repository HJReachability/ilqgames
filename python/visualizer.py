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
import matplotlib.patches as patches
from matplotlib import animation


class Visualizer(object):
    def __init__(self,
                 position_indices,
                 renderable_costs,
                 plot_lims=None,
                 figure_number=1):
        """
        Construct from list of position indices and renderable cost functions.

        :param position_indices: list of tuples of position indices (1/player)
        :type position_indices: [(uint, uint)]
        :param renderable_costs: list of cost functions that support rendering
        :type renderable_costs: [Cost]
        :param plot_lims: plot limits [xlim_low, xlim_high, ylim_low, ylim_high]
        :type plot_lims: [float, float, float, float]
        :param figure_number: which figure number to operate on
        :type figure_number: uint
        """
        self._position_indices = position_indices
        self._renderable_costs = renderable_costs
        self._figure_number = figure_number
        self._plot_lims = plot_lims

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

    def plot(self, show_last_k=-1, fade_old=True):
        """ Plot everything. """
        plt.figure(self._figure_number)
        plt.rc("text", usetex=True)

        ax = plt.gca()
        ax.set_xlabel("$x(t)$")
        ax.set_ylabel("$y(t)$")

        if self._plot_lims is not None:
            ax.set_xlim(self._plot_lims[0], self._plot_lims[1])
            ax.set_ylim(self._plot_lims[2], self._plot_lims[3])

        ax.set_aspect("equal")

        # Plot the obstacles.
        for center, radius in zip(self._obs_centers, self._obs_radii):
            circle = plt.Circle(
                (center.x, center.y), radius, color='r', fill=True, alpha=0.75)
            ax.add_artist(circle)
            ax.text(center.x - 1.25, center.y - 1.25, "obs", fontsize=8)

        # Plot the goal.
        circle = plt.Circle(
            (self._goal_center.x, self._goal_center.y),
            1, color='g', fill=True, alpha=0.75)
        ax.add_artist(circle)
        ax.text(self._goal_center.x + 1.5,
                self._goal_center.y + 1.5, "goal", fontsize=10)

        # Plot the history of trajectories.
        if show_last_k < 0 or show_last_k >= len(self._history):
            show_last_k = len(self._history)

        iterations = []

        for kk in range(len(self._history) - show_last_k, len(self._history)):
            traj = self._history[kk]
            ii = self._iterations[kk]
            xs = [x[self._x_idx, 0] for x in traj["xs"]]
            ys = [x[self._y_idx, 0] for x in traj["xs"]]

            alpha = 1
            if fade_old:
                alpha = 1 - (len(self._history) - kk) / show_last_k

            iterations.append(ii)
            plt.plot(xs, ys, 'b', label="Iteration " + str(ii), alpha=alpha)


        plt.title("ILQ solver solution (iterations {}-{})".format(
            iterations[0], iterations[-1]))

        # plt.legend()

#        plt.figure(self._figure_number + 1)
#        for ii, traj in zip(self._iterations, self._history):
#            plt.plot([np.linalg.norm(u1) for u1 in traj["u1s"]], label=str(ii) + ": u1")

#        plt.legend()

#        plt.figure(self._figure_number + 2)
#        for ii, traj in zip(self._iterations, self._history):
#            plt.plot([np.linalg.norm(u2) for u2 in traj["u2s"]], label=str(ii) + ": u2")

#        plt.legend()
