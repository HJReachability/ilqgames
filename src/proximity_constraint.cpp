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
// (Time-invariant) proximity (inequality) constraint between two vehicles, i.e.
//           g(x) = (+/-) (||(px1, py1) - (px2, py2)|| - d) <= 0
//
// NOTE: The `keep_within` argument specifies the sign of g (true corresponds to
// positive).
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/constraint/proximity_constraint.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <memory>
#include <string>

namespace ilqgames {

float ProximityConstraint::Evaluate(const VectorXf& input) const {
  const float dx = input(xidx1_) - input(xidx2_);
  const float dy = input(yidx1_) - input(yidx2_);
  const float value = std::hypot(dx, dy) - threshold_;

  return (keep_within_) ? value : -value;
}

void ProximityConstraint::Quadraticize(Time t, const VectorXf& input,
                                       MatrixXf* hess, VectorXf* grad) const {
  CHECK_NOTNULL(hess);
  CHECK_NOTNULL(grad);
  CHECK_EQ(hess->rows(), input.size());
  CHECK_EQ(hess->cols(), input.size());
  CHECK_EQ(grad->size(), input.size());

  // Compute proximity.
  const float dx = input(xidx1_) - input(xidx2_);
  const float dy = input(yidx1_) - input(yidx2_);
  const float prox = std::hypot(dx, dy);
  const float sign = (keep_within_) ? 1.0 : -1.0;
  const float g = sign * (prox - threshold_);

  // Compute gradient and Hessian.
  const float rel_dx = dx / prox;
  const float rel_dy = dy / prox;

  float grad_x1 = sign * rel_dx;
  float grad_y1 = sign * rel_dy;
  float hess_x1x1 = sign * (1.0 - rel_dx * rel_dx) / prox;
  float hess_y1y1 = sign * (1.0 - rel_dy * rel_dy) / prox;
  float hess_x1y1 = -sign * rel_dx * rel_dy / prox;

  ModifyDerivatives(t, g, &grad_x1, &hess_x1x1, &grad_y1, &hess_y1y1,
                    &hess_x1y1);

  (*grad)(xidx1_) += grad_x1;
  (*grad)(xidx2_) -= grad_x1;

  (*grad)(yidx1_) += grad_y1;
  (*grad)(yidx2_) -= grad_y1;

  (*hess)(xidx1_, xidx1_) += hess_x1x1;
  (*hess)(xidx1_, xidx2_) -= hess_x1x1;
  (*hess)(xidx2_, xidx1_) -= hess_x1x1;
  (*hess)(xidx2_, xidx2_) += hess_x1x1;

  (*hess)(yidx1_, yidx1_) += hess_y1y1;
  (*hess)(yidx1_, yidx2_) -= hess_y1y1;
  (*hess)(yidx2_, yidx1_) -= hess_y1y1;
  (*hess)(yidx2_, yidx2_) += hess_y1y1;

  (*hess)(xidx1_, yidx1_) += hess_x1y1;
  (*hess)(xidx1_, yidx2_) -= hess_x1y1;
  (*hess)(xidx2_, yidx1_) -= hess_x1y1;
  (*hess)(xidx2_, yidx2_) += hess_x1y1;
  (*hess)(yidx1_, xidx1_) += hess_x1y1;
  (*hess)(yidx1_, xidx2_) -= hess_x1y1;
  (*hess)(yidx2_, xidx1_) -= hess_x1y1;
  (*hess)(yidx2_, xidx2_) += hess_x1y1;
}

}  // namespace ilqgames
