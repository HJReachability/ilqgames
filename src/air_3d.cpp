/*
 * Copyright (c) 2020, The Regents of the University of California (Regents).
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
// Air3D dynamics, from:
// https://www.cs.ubc.ca/~mitchell/Papers/publishedIEEEtac05.pdf.
//
// Here, two Dubins cars are navigating in relative coordinates, and the usual
// setup is a pursuit-evasion game.
//
// Dynamics are:
//                 \dot r_x = -v_e + v_p cos(r_theta) + u_e r_y
//                 \dot r_y = v_p sin(r_theta) - u_e r_x
//                 \dot r_theta = u_p - u_e
// and the convention below is that controls are "omega" and the evader is P1
// and the pursuer is P2.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/dynamics/air_3d.h>

namespace ilqgames {

// Constexprs for state indices.
const Dimension Air3D::kNumXDims = 3;
const Dimension Air3D::kRxIdx = 0;
const Dimension Air3D::kRyIdx = 1;
const Dimension Air3D::kRThetaIdx = 2;

// Constexprs for control indices.
const PlayerIndex Air3D::kNumPlayers = 2;

const Dimension Air3D::kNumU1Dims = 1;
const Dimension Air3D::kOmega1Idx = 0;

const Dimension Air3D::kNumU2Dims = 1;
const Dimension Air3D::kOmega2Idx = 0;
}
