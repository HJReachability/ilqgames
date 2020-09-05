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
// Single player dynamics modeling a 2D point mass. 4 states, 2 control inputs.
// State is [x, y, xdot, ydot], control is [ax, ay], and dynamics are:
//                     \dot px    = vx
//                     \dot py    = vy
//                     \dot vx    = ax
//                     \dot vy    = ay
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/dynamics/single_player_point_mass_2d.h>

namespace ilqgames {

// Constexprs for state indices.
const Dimension SinglePlayerPointMass2D::kNumXDims = 4;
const Dimension SinglePlayerPointMass2D::kPxIdx = 0;
const Dimension SinglePlayerPointMass2D::kPyIdx = 1;
const Dimension SinglePlayerPointMass2D::kVxIdx = 2;
const Dimension SinglePlayerPointMass2D::kVyIdx = 3;

// Constexprs for control indices.
const Dimension SinglePlayerPointMass2D::kNumUDims = 2;
const Dimension SinglePlayerPointMass2D::kAxIdx = 0;
const Dimension SinglePlayerPointMass2D::kAyIdx = 1;

}  // namespace ilqgames
