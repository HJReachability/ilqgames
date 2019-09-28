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
// Two player dynamics modeling a unicycle with velocity disturbance.
// State is [x, y, theta, v], u1 is [omega, a], u2 is [dx, dy] and dynamics are:
//                     \dot px    = v cos theta + dx
//                     \dot py    = v sin theta + dy
//                     \dot theta = omega
//                     \dot v     = a
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/dynamics/two_player_unicycle_4d.h>

namespace ilqgames {

// Constexprs for state indices.
const Dimension TwoPlayerUnicycle4D::kNumXDims = 4;
const Dimension TwoPlayerUnicycle4D::kPxIdx = 0;
const Dimension TwoPlayerUnicycle4D::kPyIdx = 1;
const Dimension TwoPlayerUnicycle4D::kThetaIdx = 2;
const Dimension TwoPlayerUnicycle4D::kVIdx = 3;

// Constexprs for control indices.
const PlayerIndex TwoPlayerUnicycle4D::kNumPlayers = 2;

const Dimension TwoPlayerUnicycle4D::kNumU1Dims = 2;
const Dimension TwoPlayerUnicycle4D::kOmegaIdx = 0;
const Dimension TwoPlayerUnicycle4D::kAIdx = 1;

const Dimension TwoPlayerUnicycle4D::kNumU2Dims = 2;
const Dimension TwoPlayerUnicycle4D::kDxIdx = 0;
const Dimension TwoPlayerUnicycle4D::kDyIdx = 1;

}