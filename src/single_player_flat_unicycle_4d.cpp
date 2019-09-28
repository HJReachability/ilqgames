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
// Single player dynamics modeling a unicycle. 4 states and 2 control inputs.
// State is [x, y, theta, v], control is [omega, a], and dynamics are:
//                     \dot px    = v cos theta
//                     \dot py    = v sin theta
//                     \dot theta = omega
//                     \dot v     = a
//
//  Linear system state xi is laid out as [x, y, vx, vy]:
//                     vx = v * cos(theta)
//                     vy = v * sin(theta)
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/dynamics/single_player_flat_unicycle_4d.h>

namespace ilqgames {

// Constexprs for state indices.
const Dimension SinglePlayerFlatUnicycle4D::kNumXDims = 4;
const Dimension SinglePlayerFlatUnicycle4D::kPxIdx = 0;
const Dimension SinglePlayerFlatUnicycle4D::kPyIdx = 1;
const Dimension SinglePlayerFlatUnicycle4D::kThetaIdx = 2;
const Dimension SinglePlayerFlatUnicycle4D::kVIdx = 3;
const Dimension SinglePlayerFlatUnicycle4D::kVxIdx = 2;
const Dimension SinglePlayerFlatUnicycle4D::kVyIdx = 3;

// Constexprs for control indices.
const Dimension SinglePlayerFlatUnicycle4D::kNumUDims = 2;
const Dimension SinglePlayerFlatUnicycle4D::kOmegaIdx = 0;
const Dimension SinglePlayerFlatUnicycle4D::kAIdx = 1;

}
