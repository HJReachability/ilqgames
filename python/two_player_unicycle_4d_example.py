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

import numpy as np
import matplotlib.pyplot as plt

from two_player_unicycle_4d import TwoPlayerUnicycle4D
from ilq_solver import ILQSolver
from point import Point
from proximity_cost import ProximityCost
from semiquadratic_cost import SemiquadraticCost
from player_cost import PlayerCost
from box_constraint import BoxConstraint
from visualizer import Visualizer
from logger import Logger
import os

# General parameters.
TIME_HORIZON = 10.0   # s
TIME_RESOLUTION = 0.1 # s
HORIZON_STEPS = int(TIME_HORIZON / TIME_RESOLUTION)
LOG_DIRECTORY = './logs'

# Create dynamics.
dynamics = TwoPlayerUnicycle4D(T=0.1)

# Choose an initial state and control laws.
theta0 = np.pi / 4.0 # 45 degree heading
v0 = 10.0            # 10 m/s initial speed
x0 = np.array([[0.0],
               [0.0],
               [theta0],
               [v0]])

P1s = [np.zeros((dynamics._u1_dim, dynamics._x_dim))] * HORIZON_STEPS
P2s = [np.zeros((dynamics._u2_dim, dynamics._x_dim))] * HORIZON_STEPS
alpha1s = [np.zeros((dynamics._u1_dim, 1))] * HORIZON_STEPS
alpha2s = [np.zeros((dynamics._u2_dim, 1))] * HORIZON_STEPS

# Create the example environment. It will have a couple of circular obstacles
# laid out like this:
#                           x goal
#
#                      ()
#               ()
#                            ()
#
#          x start
goal = Point(125.0, 100.0)
obstacle_centers = [Point(40.0, 85.0), Point(80.0, 110.0), Point(100.0, 65.0)]
obstacle_radii = [10.0, 10.0, 10.0]

goal_cost = ProximityCost(
    position_indices=(0, 1), point=goal, max_distance=np.inf, name="goal")
obstacle_costs = [ProximityCost(
    position_indices=(0, 1), point=p, max_distance=r,
    name="obstacle_%f_%f" % (p.x, p.y))
                  for p, r in zip(obstacle_centers, obstacle_radii)]

# Control costs for both players to keep control in a box.
max_w = 1.0 # rad/s
max_a = 2.0 # m/s/s

w_cost_upper = SemiquadraticCost(
    dimension=0, threshold=max_w, oriented_right=True, name="w_cost_upper")
w_cost_lower = SemiquadraticCost(
    dimension=0, threshold=-max_w, oriented_right=False, name="w_cost_lower")

a_cost_upper = SemiquadraticCost(
    dimension=1, threshold=max_a, oriented_right=True, name="a_cost_upper")
a_cost_lower = SemiquadraticCost(
    dimension=1, threshold=-max_a, oriented_right=False, name="a_cost_lower")

u1_lower = np.array([[-max_w], [-max_a]])
u1_upper = np.array([[max_w], [max_a]])
u1_constraint = BoxConstraint(u1_lower, u1_upper)

max_dvx = 0.1 # m/s
max_dvy = 0.1 # m/s

dvx_cost_upper = SemiquadraticCost(
    dimension=0, threshold=max_dvx, oriented_right=True, name="dvx_cost_upper")
dvx_cost_lower = SemiquadraticCost(
    dimension=0, threshold=-max_dvx, oriented_right=False, name="dvx_cost_lower")

dvy_cost_upper = SemiquadraticCost(
    dimension=1, threshold=max_dvy, oriented_right=True, name="dvy_cost_upper")
dvy_cost_lower = SemiquadraticCost(
    dimension=1, threshold=-max_dvy, oriented_right=False, name="dvy_cost_lower")

u2_lower = np.array([[-max_dvx], [-max_dvy]])
u2_upper = np.array([[max_dvx], [max_dvy]])
u2_constraint = BoxConstraint(u2_lower, u2_upper)

# Add light quadratic from origin for controls.
light_cost_upper0 = SemiquadraticCost(
    dimension=0, threshold=-0.01, oriented_right=True, name="light_cost_upper0")
light_cost_lower0 = SemiquadraticCost(
    dimension=0, threshold=-0.01, oriented_right=False, name="light_cost_lower0")

light_cost_upper1 = SemiquadraticCost(
    dimension=1, threshold=-0.01, oriented_right=True, name="light_cost_upper1")
light_cost_lower1 = SemiquadraticCost(
    dimension=1, threshold=-0.01, oriented_right=False, name="light_cost_lower1")

# Add light quadratic around original values for theta/v.
theta_light_cost_upper = SemiquadraticCost(
    dimension=2, threshold=theta0-0.01, oriented_right=True, name="theta_light_cost_upper")
theta_light_cost_lower = SemiquadraticCost(
    dimension=2, threshold=theta0-0.01, oriented_right=False, name="theta_light_cost_lower")

v_light_cost_upper = SemiquadraticCost(
    dimension=3, threshold=v0-0.01, oriented_right=True, name="v_light_cost_upper")
v_light_cost_lower = SemiquadraticCost(
    dimension=3, threshold=v0-0.01, oriented_right=False, name="v_light_cost_lower")

# Build up total costs for both players. This is basically a zero-sum game.
player1_cost = PlayerCost()
player1_cost.add_cost(goal_cost, "x", -1.0)
for cost in obstacle_costs:
    player1_cost.add_cost(cost, "x", 10.0)

player1_cost.add_cost(theta_light_cost_lower, "x", 0.1)
player1_cost.add_cost(theta_light_cost_upper, "x", 0.1)
player1_cost.add_cost(v_light_cost_lower, "x", 0.1)
player1_cost.add_cost(v_light_cost_upper, "x", 0.1)

player1_cost.add_cost(w_cost_upper, "u1", 10.0)
player1_cost.add_cost(w_cost_lower, "u1", 10.0)
player1_cost.add_cost(a_cost_upper, "u1", 10.0)
player1_cost.add_cost(a_cost_lower, "u1", 10.0)

player1_cost.add_cost(light_cost_upper0, "u1", 1.0)
player1_cost.add_cost(light_cost_lower0, "u1", 1.0)
player1_cost.add_cost(light_cost_upper1, "u1", 1.0)
player1_cost.add_cost(light_cost_lower1, "u1", 1.0)

player2_cost = PlayerCost()
player2_cost.add_cost(goal_cost, "x", 1.0)
for cost in obstacle_costs:
    player2_cost.add_cost(cost, "x", -10.0)

player2_cost.add_cost(theta_light_cost_lower, "x", 0.1)
player2_cost.add_cost(theta_light_cost_upper, "x", 0.1)
player2_cost.add_cost(v_light_cost_lower, "x", 0.1)
player2_cost.add_cost(v_light_cost_upper, "x", 0.1)

player2_cost.add_cost(dvx_cost_upper, "u2", 10.0)
player2_cost.add_cost(dvx_cost_lower, "u2", 10.0)
player2_cost.add_cost(dvy_cost_upper, "u2", 10.0)
player2_cost.add_cost(dvy_cost_lower, "u2", 10.0)

player2_cost.add_cost(light_cost_upper0, "u2", 1.0)
player2_cost.add_cost(light_cost_lower0, "u2", 1.0)
player2_cost.add_cost(light_cost_upper1, "u2", 1.0)
player2_cost.add_cost(light_cost_lower1, "u2", 1.0)

# Visualizer.
visualizer = Visualizer(0, 1, obstacle_centers, obstacle_radii, goal)

# Logger.
if not os.path.exists(LOG_DIRECTORY):
    os.makedirs(LOG_DIRECTORY)

logger = Logger(os.path.join(LOG_DIRECTORY, 'unicycle_4d_example.pkl'))

# Set up ILQSolver.
solver = ILQSolver(dynamics, player1_cost, player2_cost,
                   x0, P1s, P2s, alpha1s, alpha2s,
                   u1_constraint, u2_constraint,
                   logger, visualizer)

solver.run()
