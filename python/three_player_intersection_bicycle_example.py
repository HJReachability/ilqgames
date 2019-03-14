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
# Script to run a 3 player collision avoidance example intended to model
# a T-intersection.
#
################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt

from unicycle_4d import Unicycle4D
from bicycle_4d import Bicycle4D
from product_multiplayer_dynamical_system import \
    ProductMultiPlayerDynamicalSystem

from point import Point
from polyline import Polyline

from ilq_solver import ILQSolver
from proximity_cost import ProximityCost
from product_state_proximity_cost import ProductStateProximityCost
from semiquadratic_cost import SemiquadraticCost
from quadratic_cost import QuadraticCost
from semiquadratic_polyline_cost import SemiquadraticPolylineCost
from quadratic_polyline_cost import QuadraticPolylineCost
from player_cost import PlayerCost
from box_constraint import BoxConstraint

from visualizer import Visualizer
from logger import Logger

# General parameters.
TIME_HORIZON = 5.0   # s
TIME_RESOLUTION = 0.25 # s
HORIZON_STEPS = int(TIME_HORIZON / TIME_RESOLUTION)
LOG_DIRECTORY = "./logs/three_player/"

# Create dynamics.
car1 = Unicycle4D()
car2 = Unicycle4D()
bike = Bicycle4D(0.5, 0.5)
dynamics = ProductMultiPlayerDynamicalSystem(
    [car1, car2, bike], T=TIME_RESOLUTION)

# Choose initial states and set initial control laws to zero, such that
# we start with a situation that looks like this:
#
#              (car 2)
#             |   X   .       |
#             |   :   .       |
#             |  \./  .       |
#             |       .      <--X (bike)
#             |       .        ------------------
#             |       .
#             |       .        ..................
#             |       .
#             |       .        ------------------
#             |       .   ^   |
#             |       .   :   |         (+y)
#             |       .   :   |          |
#             |       .   X   |          |
#                      (car 1)           |______ (+x)
#
# We shall set up the costs so that car 2 wants to turn and car 1 / bike 1
# continue straight in their initial direction of motion.
# We shall assume that lanes are 4 m wide and set the origin to be in the
# bottom left along the road boundary.
car1_theta0 = np.pi / 2.0 # 90 degree heading
car1_v0 = 1.0             # 5 m/s initial speed
car1_x0 = np.array([
    [6.5],
    [0.0],
    [car1_theta0],
    [car1_v0]
])

car2_theta0 = -np.pi / 2.0 # -90 degree heading
car2_v0 = 0.1              # 2 m/s initial speed
car2_x0 = np.array([
    [1.5],
    [40.0],
    [car2_theta0],
    [car2_v0]
])

bike_psi0 = 0.0 # moving right
bike_v0 = 2.0   # 0.1 m/s initial speed
bike_x0 = np.array([
    [0.0],
    [22.0],
    [bike_psi0],
    [bike_v0]
])

stacked_x0 = np.concatenate([car1_x0, car2_x0, bike_x0], axis=0)

car1_Ps = [np.zeros((car1._u_dim, dynamics._x_dim))] * HORIZON_STEPS
car2_Ps = [np.zeros((car2._u_dim, dynamics._x_dim))] * HORIZON_STEPS
bike_Ps = [np.zeros((bike._u_dim, dynamics._x_dim))] * HORIZON_STEPS

car1_alphas = [np.zeros((car1._u_dim, 1))] * HORIZON_STEPS
car2_alphas = [np.zeros((car2._u_dim, 1))] * HORIZON_STEPS
bike_alphas = [np.zeros((bike._u_dim, 1))] * HORIZON_STEPS

# Create environment.
car1_position_indices_in_product_state = (0, 1)
car1_polyline = Polyline([Point(6.0, -100.0), Point(6.0, 100.0)])
car1_polyline_boundary_cost = SemiquadraticPolylineCost(
    car1_polyline, 1.0, car1_position_indices_in_product_state,
    "car1_polyline_boundary")
car1_polyline_cost = QuadraticPolylineCost(
    car1_polyline, car1_position_indices_in_product_state, "car1_polyline")

car1_goal = Point(6.0, 35.0)
car1_goal_cost = ProximityCost(
    car1_position_indices_in_product_state, car1_goal, np.inf, "car1_goal")

car2_position_indices_in_product_state = (4, 5)
car2_polyline = Polyline([Point(2.0, 100.0),
                          Point(2.0, 18.0),
                          Point(2.5, 15.0),
                          Point(3.0, 14.0),
                          Point(5.0, 12.5),
                          Point(8.0, 12.0),
                          Point(100.0, 12.0)])
car2_polyline_boundary_cost = SemiquadraticPolylineCost(
    car2_polyline, 1.0, car2_position_indices_in_product_state,
    "car2_polyline_boundary")
car2_polyline_cost = QuadraticPolylineCost(
    car2_polyline, car2_position_indices_in_product_state, "car2_polyline")

car2_goal = Point(12.0, 12.0)
car2_goal_cost = ProximityCost(
    car2_position_indices_in_product_state, car2_goal, np.inf, "car2_goal")

bike_position_indices_in_product_state = (8, 9)
bike_goal = Point(15.0, 21.0)
bike_goal_cost = ProximityCost(
    bike_position_indices_in_product_state, bike_goal, np.inf, "bike_goal")

# Penalize speed above a threshold for all players.
car1_v_index_in_product_state = 3
car1_maxv = 10.0 # m/s
car1_minv_cost = SemiquadraticCost(
    car1_v_index_in_product_state, 0.0, False, "car1_minv")
car1_maxv_cost = SemiquadraticCost(
    car1_v_index_in_product_state, car1_maxv, True, "car1_maxv")

car2_v_index_in_product_state = 7
car2_maxv = 10.0 # m/s
car2_minv_cost = SemiquadraticCost(
    car2_v_index_in_product_state, 0.0, False, "car2_minv")
car2_maxv_cost = SemiquadraticCost(
    car2_v_index_in_product_state, car2_maxv, True, "car2_maxv")

bike_psi_index_in_product_state = 10
bike_v_index_in_product_state = 11
bike_maxv = 2.5 # m/s
bike_minv_cost = SemiquadraticCost(
    bike_v_index_in_product_state, 1.0, False, "bike_minv")
bike_maxv_cost = SemiquadraticCost(
    bike_v_index_in_product_state, bike_maxv, True, "bike_maxv")

# Control costs for all players.
car1_w_cost = QuadraticCost(0, 0.0, "car1_w_cost")
car1_a_cost = QuadraticCost(1, 0.0, "car1_a_cost")

car2_w_cost = QuadraticCost(0, 0.0, "car2_w_cost")
car2_a_cost = QuadraticCost(1, 0.0, "car2_a_cost")

bike_deltaf_cost = QuadraticCost(0, 0.0, "bike_deltaf_cost")
bike_deltaf_barrier_upper = SemiquadraticCost(
    0, np.pi / 4.0, True, "bike_deltaf_upper")
bike_deltaf_barrier_lower = SemiquadraticCost(
    0, -np.pi / 4.0, False, "bike_deltaf_lower")
bike_a_cost = QuadraticCost(1, 0.0, "bike_a_cost")

# Proximity cost.
CAR_PROXIMITY_THRESHOLD = 2.0
BIKE_PROXIMITY_THRESHOLD = 1.0
car1_proximity_cost = ProductStateProximityCost(
    [car1_position_indices_in_product_state,
     car2_position_indices_in_product_state,
     bike_position_indices_in_product_state],
    CAR_PROXIMITY_THRESHOLD,
    "car1_proximity")
car2_proximity_cost = ProductStateProximityCost(
    [car1_position_indices_in_product_state,
     car2_position_indices_in_product_state,
     bike_position_indices_in_product_state],
    CAR_PROXIMITY_THRESHOLD,
    "car2_proximity")
bike_proximity_cost = ProductStateProximityCost(
    [car1_position_indices_in_product_state,
     car2_position_indices_in_product_state,
     bike_position_indices_in_product_state],
    BIKE_PROXIMITY_THRESHOLD,
    "bike_proximity")

# Build up total costs for both players. This is basically a zero-sum game.
car1_cost = PlayerCost()
car1_cost.add_cost(car1_goal_cost, "x", -1.0)
car1_cost.add_cost(car1_polyline_cost, "x", 50.0)
car1_cost.add_cost(car1_polyline_boundary_cost, "x", 50.0)
car1_cost.add_cost(car1_maxv_cost, "x", 100.0)
car1_cost.add_cost(car1_minv_cost, "x", 100.0)
car1_cost.add_cost(car1_proximity_cost, "x", 100.0)

car1_player_id = 0
car1_cost.add_cost(car1_w_cost, car1_player_id, 25.0)
car1_cost.add_cost(car1_a_cost, car1_player_id, 1.0)

car2_cost = PlayerCost()
car2_cost.add_cost(car2_goal_cost, "x", -1.0)
car2_cost.add_cost(car2_polyline_cost, "x", 50.0)
car2_cost.add_cost(car2_polyline_boundary_cost, "x", 50.0)
car2_cost.add_cost(car2_maxv_cost, "x", 100.0)
car2_cost.add_cost(car2_minv_cost, "x", 100.0)
car2_cost.add_cost(car2_proximity_cost, "x", 100.0)

car2_player_id = 1
car2_cost.add_cost(car2_w_cost, car2_player_id, 25.0)
car2_cost.add_cost(car2_a_cost, car2_player_id, 1.0)

bike_cost = PlayerCost()
bike_cost.add_cost(bike_goal_cost, "x", -1.0)
bike_cost.add_cost(bike_maxv_cost, "x", 100.0)
bike_cost.add_cost(bike_minv_cost, "x", 100.0)
bike_cost.add_cost(bike_proximity_cost, "x", 1.0)

bike_player_id = 2
bike_cost.add_cost(bike_deltaf_cost, bike_player_id, 1.0)
#bike_cost.add_cost(bike_deltaf_barrier_lower, bike_player_id, 100.0)
#bike_cost.add_cost(bike_deltaf_barrier_upper, bike_player_id, 100.0)
bike_cost.add_cost(bike_a_cost, bike_player_id, 1.0)

# Visualizer.
visualizer = Visualizer(
    [car1_position_indices_in_product_state,
     car2_position_indices_in_product_state,
     bike_position_indices_in_product_state],
    [car1_polyline_boundary_cost,
     car1_goal_cost,
     car2_polyline_boundary_cost,
     car2_goal_cost,
     bike_goal_cost],
    [".-r", ".-g", ".-b"],
    1,
    False,
    plot_lims=[-5, 25, -5, 45])

# Logger.
if not os.path.exists(LOG_DIRECTORY):
    os.makedirs(LOG_DIRECTORY)

logger = Logger(os.path.join(LOG_DIRECTORY, 'intersection_bicycle_example.pkl'))

# Set up ILQSolver.
solver = ILQSolver(dynamics,
                   [car1_cost, car2_cost, bike_cost],
                   stacked_x0,
                   [car1_Ps, car2_Ps, bike_Ps],
                   [car1_alphas, car2_alphas, bike_alphas],
                   0.1,
                   None,
                   logger,
                   visualizer,
                   None)

solver.run()
