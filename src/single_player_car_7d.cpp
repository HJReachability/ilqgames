/*
 * Copyright (c) 2019, The Regents of the University of California (Regents).
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *    1. Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *    2. Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 *
 *    3. Neither the name of the copyright holder nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Please contact the author(s) of this library if you have any questions.
 * Authors: David Fridovich-Keil   ( dfk@eecs.berkeley.edu )
 */

///////////////////////////////////////////////////////////////////////////////
//
// Single player dynamics modeling a car. 6 states and 2 control inputs.
// State is [x, y, theta, phi, v, kappa, s], control is [omega, a], and dynamics
// are:
//                     \dot px    = v cos theta
//                     \dot py    = v sin theta
//                     \dot theta = (v / L) * tan phi
//                     \dot phi   = omega
//                     \dot v     = a
//                     \dot kappa = omega / (L * cos^2 phi)   [ curvature ]
//                     \dot s     = v                         [ path length ]
// Please refer to
// https://www.sciencedirect.com/science/article/pii/S2405896316301215
// for further details.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/dynamics/single_player_car_7d.h>

namespace ilqgames {

// Constexprs for state indices.
const Dimension SinglePlayerCar7D::kNumXDims = 7;
const Dimension SinglePlayerCar7D::kPxIdx = 0;
const Dimension SinglePlayerCar7D::kPyIdx = 1;
const Dimension SinglePlayerCar7D::kThetaIdx = 2;
const Dimension SinglePlayerCar7D::kPhiIdx = 3;
const Dimension SinglePlayerCar7D::kVIdx = 4;
const Dimension SinglePlayerCar7D::kKappaIdx = 5;
const Dimension SinglePlayerCar7D::kSIdx = 6;

// Constexprs for control indices.
const Dimension SinglePlayerCar7D::kNumUDims = 2;
const Dimension SinglePlayerCar7D::kOmegaIdx = 0;
const Dimension SinglePlayerCar7D::kAIdx = 1;

}  // namespace ilqgames
