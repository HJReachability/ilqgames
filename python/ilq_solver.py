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
# Iterative LQ solver.
#
################################################################################

import numpy as np
import torch
import matplotlib.pyplot as plt

from player_cost import PlayerCost
from reference_deviation_cost import ReferenceDeviationCost
from solve_lq_game import solve_lq_game
from visualizer import Visualizer
from logger import Logger

class ILQSolver(object):
    def __init__(self,
                 dynamics,
                 player_costs,
                 x0,
                 Ps,
                 alphas,
                 alpha_scaling=0.05,
                 reference_deviation_weight=None,
                 logger=None,
                 visualizer=None,
                 u_constraints=None):
        """
        Initialize from dynamics, player costs, current state, and initial
        guesses for control strategies for both players.

        :param dynamics: two-player dynamical system
        :type dynamics: TwoPlayerDynamicalSystem
        :param player_costs: list of cost functions for all players
        :type player_costs: [PlayerCost]
        :param x0: initial state
        :type x0: np.array
        :param Ps: list of lists of feedback gains (1 list per player)
        :type Ps: [[np.array]]
        :param alphas: list of lists of feedforward terms (1 list per player)
        :type alphas: [[np.array]]
        :param alpha_scaling: step size on the alpha
        :type alpha_scaling: float
        :param reference_deviation_weight: weight on reference deviation cost
        :type reference_deviation_weight: None or float
        :param logger: logging utility
        :type logger: Logger
        :param visualizer: optional visualizer
        :type visualizer: Visualizer
        :param u_constraints: list of constraints on controls
        :type u_constraints: [Constraint]
        """
        self._dynamics = dynamics
        self._player_costs = player_costs
        self._x0 = x0
        self._Ps = Ps
        self._alphas = alphas
        self._u_constraints = u_constraints
        self._horizon = len(Ps[0])
        self._num_players = len(player_costs)

        # Current and previous operating points (states/controls) for use
        # in checking convergence.
        self._last_operating_point = None
        self._current_operating_point = None

        # Fixed step size for the linesearch.
        self._alpha_scaling = alpha_scaling

        # Reference deviation cost weight.
        self._reference_deviation_weight = reference_deviation_weight

        # Set up visualizer.
        self._visualizer = visualizer
        self._logger = logger

        # Log some of the paramters.
        if self._logger is not None:
            self._logger.log("alpha_scaling", self._alpha_scaling)
            self._logger.log("horizon", self._horizon)
            self._logger.log("x0", self._x0)

    def run(self):
        """ Run the algorithm for the specified parameters. """
        iteration = 0

        while not self._is_converged():
            # (1) Compute current operating point and update last one.
            xs, us, costs = self._compute_operating_point()
            self._last_operating_point = self._current_operating_point
            self._current_operating_point = (xs, us, costs)

            # If this is the first time through, then set up reference deviation
            # costs and add to player costs. Otherwise, just update those costs.
            if self._reference_deviation_weight is not None and iteration == 0:
                self._x_reference_cost = ReferenceDeviationCost(xs)
                self._u_reference_costs = [
                    ReferenceDeviationCost(ui) for ui in us]

                for ii in range(self._num_players):
                    self._player_costs[ii].add_cost(
                        self._x_reference_cost, "x",
                        self._reference_deviation_weight)
                    self._player_costs[ii].add_cost(
                        self._u_reference_costs[ii], ii,
                        self._reference_deviation_weight)
            elif self._reference_deviation_weight is not None:
                self._x_reference_cost.reference = self._last_operating_point[0]
                for ii in range(self._num_players):
                    self._u_reference_costs[ii].reference = \
                        self._last_operating_point[1][ii]

            # Visualization.
            if self._visualizer is not None:
                traj = {"xs" : xs}
                for ii in range(self._num_players):
                    traj["u%ds" % (ii + 1)] = us[ii]

                self._visualizer.add_trajectory(iteration, traj)
#                self._visualizer.plot_controls(1)
#                plt.pause(0.01)
#                plt.clf()
#                self._visualizer.plot_controls(2)
#                plt.pause(0.01)
#                plt.clf()
                self._visualizer.plot()
                plt.pause(0.01)
                plt.clf()

            # (2) Linearize about this operating point. Make sure to
            # stack appropriately since we will concatenate state vectors
            # but not control vectors, so that
            #    ``` x_{k+1} - xs_k = A_k (x_k - xs_k) +
            #          sum_i Bi_k (ui_k - uis_k) ```
            As = []
            Bs = [[] for ii in range(self._num_players)]
            for k in range(self._horizon):
                A, B = self._dynamics.linearize_discrete(
                    xs[k], [uis[k] for uis in us])
                As.append(A)

                for ii in range(self._num_players):
                    Bs[ii].append(B[ii])

            # (3) Quadraticize costs.
            Qs = [[] for ii in range(self._num_players)]
            ls = [[] for ii in range(self._num_players)]
            Rs = [[[] for jj in range(self._num_players)]
                  for ii in range(self._num_players)]
            for ii in range(self._num_players):
                for k in range(self._horizon):
                    _, l, Q, R = self._player_costs[ii].quadraticize(
                        xs[k], [uis[k] for uis in us], k)

                    Qs[ii].append(Q)
                    ls[ii].append(l)

                    for jj in range(self._num_players):
                        Rs[ii][jj].append(R[jj])

            # (4) Compute feedback Nash equilibrium of the resulting LQ game.
            Ps, alphas = solve_lq_game(As, Bs, Qs, ls, Rs)

            # Accumulate total costs for both players.
            total_costs = [sum(costis).item() for costis in costs]
            print("Total cost for all players: ", total_costs)

            # Log everything.
            if self._logger is not None:
                self._logger.log("xs", xs)
                self._logger.log("us", us)
                self._logger.log("total_costs", total_costs)
                self._logger.dump()

            # Update the member variables.
            self._Ps = Ps
            self._alphas = alphas

            # (5) Linesearch.
            self._linesearch()
            iteration += 1

    def _compute_operating_point(self):
        """
        Compute current operating point by propagating through dynamics.

        :return: states, controls for all players (list of lists), and
            costs over time (list of lists), i.e. (xs, us, costs)
        :rtype: [np.array], [[np.array]], [[torch.Tensor(1, 1)]]
        """
        xs = [self._x0]
        us = [[] for ii in range(self._num_players)]
        costs = [[] for ii in range(self._num_players)]

        for k in range(self._horizon):
            if self._current_operating_point is not None:
                current_x = self._current_operating_point[0][k]
                current_u = [self._current_operating_point[1][ii][k]
                             for ii in range(self._num_players)]
            else:
                current_x = np.zeros((self._dynamics._x_dim, 1))
                current_u = [np.zeros((ui_dim, 1))
                             for ui_dim in self._dynamics._u_dims]

            feedback = lambda x, u_ref, x_ref, P, alpha : \
                       u_ref - P @ (x - x_ref) - self._alpha_scaling * alpha
            u = [feedback(xs[k], current_u[ii], current_x,
                          self._Ps[ii][k], self._alphas[ii][k])
                 for ii in range(self._num_players)]

            # Clip u1 and u2.
#            for ii in range(self._num_players):
#                u[ii] = self._u_constraints[ii].clip(u[ii])

            for ii in range(self._num_players):
                us[ii].append(u[ii])
                costs[ii].append(self._player_costs[ii](
                    torch.as_tensor(xs[k].copy()),
                    [torch.as_tensor(ui) for ui in u],
                    k))

            if k == self._horizon - 1:
                break

            xs.append(self._dynamics.integrate(xs[k], u))

        return xs, us, costs

    def _linesearch(self):
        """ Linesearch for both players separately. """
        pass

    def _is_converged(self):
        """ Check if the last two operating points are close enough. """
        if self._last_operating_point is None:
            return False

        # Tolerance for comparing operating points. If all states changes
        # within this tolerance in the Euclidean norm then we've converged.
        TOLERANCE = 1e-4
        for ii in range(self._horizon):
            last_x = self._last_operating_point[0][ii]
            current_x = self._current_operating_point[0][ii]

            if np.linalg.norm(last_x - current_x) > TOLERANCE:
                return False

        return True
