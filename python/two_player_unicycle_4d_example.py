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
# Script to run an obstacle avoidance example for the TwoPlayerUnicycle4D.
#
################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt

from two_player_unicycle_4d import TwoPlayerUnicycle4D
from ilq_solver import ILQSolver
from point import Point
from proximity_cost import ProximityCost
from obstacle_cost import ObstacleCost
from semiquadratic_cost import SemiquadraticCost
from quadratic_cost import QuadraticCost
from player_cost import PlayerCost
from box_constraint import BoxConstraint
from visualizer import Visualizer
from logger import Logger

# General parameters.
TIME_HORIZON = 10.0   # s
TIME_RESOLUTION = 0.1 # s
HORIZON_STEPS = int(TIME_HORIZON / TIME_RESOLUTION)
LOG_DIRECTORY = "./logs/two_player_zero_sum/"
MAX_V = 15.0 # m/s

# Create dynamics.
dynamics = TwoPlayerUnicycle4D(T=TIME_RESOLUTION)

# Choose an initial state and control laws.
theta0 = np.pi / 2.5 # 60 degree heading
v0 = 5.0             # 5 m/s initial speed
x0 = np.array([[0.0],
               [0.0],
               [theta0],
               [v0]])

P1s = [np.zeros((dynamics._u_dims[0], dynamics._x_dim))] * HORIZON_STEPS
P2s = [np.zeros((dynamics._u_dims[1], dynamics._x_dim))] * HORIZON_STEPS
alpha1s = [np.zeros((dynamics._u_dims[0], 1))] * HORIZON_STEPS
alpha2s = [np.zeros((dynamics._u_dims[1], 1))] * HORIZON_STEPS

# Create the example environment. It will have a couple of circular obstacles
# laid out like this:
#                           x goal
#
#                      ()
#               ()
#                            ()
#
#          x start
goal = Point(100.0, 100.0)
obstacle_centers = [Point(100.0, 35.0),
                    Point(65.0, 65.0), Point(25.0, 80.0)]
obstacle_radii = [10.0, 10.0, 10.0]

goal_cost = ProximityCost(
    position_indices=(0, 1), point=goal, max_distance=np.inf, name="goal")
obstacle_costs = [ObstacleCost(
    position_indices=(0, 1), point=p, max_distance=r,
    name="obstacle_%f_%f" % (p.x, p.y))
                  for p, r in zip(obstacle_centers, obstacle_radii)]

# Control costs for both players to keep control in a box.
max_w = 1.0 # rad/s
max_a = 2.0 # m/s/s

w_cost = QuadraticCost(dimension=0, origin=0, name="w_cost")
a_cost = QuadraticCost(dimension=1, origin=0, name="a_cost")

dvx_cost = QuadraticCost(dimension=0, origin=0, name="dvx_cost")
dvy_cost = QuadraticCost(dimension=1, origin=0, name="dvy_cost")

# Add light quadratic around original values for theta/v.
v_cost_upper = SemiquadraticCost(
    dimension=3, threshold=MAX_V, oriented_right=True, name="v_cost_upper")
v_cost_lower = SemiquadraticCost(
    dimension=3, threshold=0, oriented_right=False, name="v_cost_lower")

OBSTACLE_WEIGHT = 100.0
GOAL_WEIGHT = 1.0

# Build up total costs for both players. This is basically a zero-sum game.
player1_cost = PlayerCost()
player1_cost.add_cost(goal_cost, "x", -GOAL_WEIGHT)
for cost in obstacle_costs:
    player1_cost.add_cost(cost, "x", OBSTACLE_WEIGHT)

player1_cost.add_cost(v_cost_upper, "x", 20.0)
player1_cost.add_cost(v_cost_lower, "x", 20.0)
player1_cost.add_cost(w_cost, 0, 10.0)
player1_cost.add_cost(a_cost, 0, 10.0)

player2_cost = PlayerCost()
player2_cost.add_cost(goal_cost, "x", GOAL_WEIGHT)
for cost in obstacle_costs:
    player2_cost.add_cost(cost, "x", -OBSTACLE_WEIGHT)

player2_cost.add_cost(v_cost_upper, "x", -20.0)
player2_cost.add_cost(v_cost_lower, "x", -20.0)
player2_cost.add_cost(dvx_cost, 1, 10.0)
player2_cost.add_cost(dvy_cost, 1, 10.0)

# Visualizer.
visualizer = Visualizer([(0, 1)], [goal_cost] + obstacle_costs, [".-b"],
                        plot_lims=[0, 175, 0, 175])

# Logger.
if not os.path.exists(LOG_DIRECTORY):
    os.makedirs(LOG_DIRECTORY)

logger = Logger(os.path.join(LOG_DIRECTORY, 'unicycle_4d_example.pkl'))

# Set up ILQSolver.
solver = ILQSolver(dynamics, [player1_cost, player2_cost],
                   x0, [P1s, P2s], [alpha1s, alpha2s],
                   0.025, logger, visualizer)

solver.run()
