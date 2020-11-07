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
// Base class for all explicit (scalar-valued) equality constraints. These
// constraints are of the form: g(x) = 0 for some vector x.
//
// In addition to checking for satisfaction (and returning the constraint value
// g(x)), they also support computing first and second derivatives of the
// constraint value itself and the square of the constraint value, each scaled
// by lambda or mu respectively (from the augmented Lagrangian). That is, they
// compute gradients and Hessians of
//         L(x, lambda, mu) = lambda * g(x) + mu * g(x) * g(x) / 2
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/constraint/constraint.h>
#include <ilqgames/utils/relative_time_tracker.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <memory>
#include <string>

namespace ilqgames {

float Constraint::mu_ = constants::kDefaultMu;

void Constraint::ModifyDerivatives(Time t, float g, float* dx, float* ddx,
                                   float* dy, float* ddy, float* dxdy) const {
  // Unpack lambda.
  const float lambda = lambdas_[TimeIndex(t)];
  const float mu = Mu(lambda, g);

  // Assumes that these are just the derivatives of g(x, y), and modifies them
  // to be derivatives of lambda g(x) + mu g(x) g(x) / 2.
  const float new_dx = lambda * *dx + mu * g * *dx;
  const float new_ddx = lambda * *ddx + mu * (*dx * *dx + g * *ddx);

  if (dy) {
    CHECK_NOTNULL(ddy);
    CHECK_NOTNULL(dxdy);

    const float new_dy = lambda * *dy + mu * g * *dy;
    const float new_ddy = lambda * *ddy + mu * (*dy * *dy + g * *ddy);
    const float new_dxdy = lambda * *dxdy + mu * (*dy * *dx + g * *dxdy);

    *dy = new_dy;
    *ddy = new_ddy;
    *dxdy = new_dxdy;
  }

  *dx = new_dx;
  *ddx = new_ddx;
}

}  // namespace ilqgames
