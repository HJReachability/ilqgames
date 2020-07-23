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
// Base class for all cost functions. All costs must support evaluation and
// quadraticization. By default, cost functions are of only state or control.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/cost/cost.h>

namespace ilqgames {

Time Cost::initial_time_ = 0.0;

void Cost::ModifyDerivatives(float exponential_constant, Time t,
                             const VectorXf& input, float* dx, float* ddx,
                             float* dy, float* ddy, float* dxdy, float* dz,
                             float* ddz, float* dxdz, float* dydz) const {
  if (!IsExponentiated() && exponential_constant == 0.0) return;

  const float scaling = (exponential_constant == 0.0)
                            ? exponential_sign_ * exponential_constant_
                            : exponential_constant;

  const float exp_cost = EvaluateExponential(t, input);
  const float modified_dx = scaling * *dx * exp_cost;
  const float modified_ddx = scaling * exp_cost * (*ddx + scaling * *dx * *dx);

  if (dy && ddy && dxdy) {
    const float modified_dy = scaling * *dy * exp_cost;
    const float modified_ddy =
        scaling * exp_cost * (*ddy + scaling * *dy * *dy);
    const float modified_dxdy =
        scaling * exp_cost * (*dxdy + scaling * *dx * *dy);

    *dy = modified_dy;
    *ddy = modified_ddy;
    *dxdy = modified_dxdy;
  }

  if (dz && ddz && dxdz) {
    const float modified_dz = scaling * *dz * exp_cost;
    const float modified_ddz =
        scaling * exp_cost * (*ddz + scaling * *dz * *dz);
    const float modified_dxdz =
        scaling * exp_cost * (*dxdz + scaling * *dx * *dz);

    *dz = modified_dz;
    *ddz = modified_ddz;
    *dxdz = modified_dxdz;
  }

  if (dz && dy) {
    const float modified_dydz =
        scaling * exp_cost * (*dydz + scaling * *dy * *dz);
    *dydz = modified_dydz;
  }

  *dx = modified_dx;
  *ddx = modified_ddx;
}
}  // namespace ilqgames
