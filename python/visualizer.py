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
                 player_linestyles,
                 show_last_k=1,
                 fade_old=False,
                 plot_lims=None,
                 figure_number=1):
        """
        Construct from list of position indices and renderable cost functions.

        :param position_indices: list of tuples of position indices (1/player)
        :type position_indices: [(uint, uint)]
        :param renderable_costs: list of cost functions that support rendering
        :type renderable_costs: [Cost]
        :param player_linestyles: list of line styles (1 per player, e.g. ".-r")
        :type player_colors: [string]
        :param show_last_k: how many of last trajectories to plot (-1 shows all)
        :type show_last_k: int
        :param fade_old: flag for fading older trajectories
        :type fade_old: bool
        :param plot_lims: plot limits [xlim_low, xlim_high, ylim_low, ylim_high]
        :type plot_lims: [float, float, float, float]
        :param figure_number: which figure number to operate on
        :type figure_number: uint
        """
        self._position_indices = position_indices
        self._renderable_costs = renderable_costs
        self._player_linestyles = player_linestyles
        self._show_last_k = show_last_k
        self._fade_old = fade_old
        self._figure_number = figure_number
        self._plot_lims = plot_lims
        self._num_players = len(position_indices)

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
        plt.rc("text", usetex=True)

        ax = plt.gca()
        ax.set_xlabel("$x(t)$")
        ax.set_ylabel("$y(t)$")

        if self._plot_lims is not None:
            ax.set_xlim(self._plot_lims[0], self._plot_lims[1])
            ax.set_ylim(self._plot_lims[2], self._plot_lims[3])

        ax.set_aspect("equal")

        # Render all costs.
        for cost in self._renderable_costs:
            cost.render(ax)

        # Plot the history of trajectories for each player.
        if self._show_last_k < 0 or self._show_last_k >= len(self._history):
            show_last_k = len(self._history)
        else:
            show_last_k = self._show_last_k

        plotted_iterations = []
        for kk in range(len(self._history) - show_last_k, len(self._history)):
            traj = self._history[kk]
            iteration = self._iterations[kk]
            plotted_iterations.append(iteration)

            alpha = 1.0
            if self._fade_old:
                alpha = 1.0 - float(len(self._history) - kk) / show_last_k

            for ii in range(self._num_players):
                x_idx, y_idx = self._position_indices[ii]
                xs = [x[x_idx, 0] for x in traj["xs"]]
                ys = [x[y_idx, 0] for x in traj["xs"]]
                plt.plot(xs, ys,
                         self._player_linestyles[ii],
                         label="Player {}, iteration {}".format(ii, iteration),
                         alpha=alpha,
                         markersize=2)

        plt.title("ILQ solver solution (iterations {}-{})".format(
            plotted_iterations[0], plotted_iterations[-1]))

    def plot_controls(self, player_number):
        """ Plot control for both players. """
        plt.figure(self._figure_number + player_number)
        uis = "u%ds" % player_number
        plt.plot([ui[0, 0] for ui in self._history[-1][uis]], "*:r", label="u1")
        plt.plot([ui[1, 0] for ui in self._history[-1][uis]], "*:b", label="u2")
        plt.legend()
        plt.title("Controls for Player %d" % player_number)
