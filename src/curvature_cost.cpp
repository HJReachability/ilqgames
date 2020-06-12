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
// Quadratic cost on curvature (angular speed / longitudinal speed)
// NOTE: this is currently implemented as a state cost.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/cost/curvature_cost.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>

namespace ilqgames {

float CurvatureCost::Evaluate(const VectorXf& input) const {
  const float curvature = Curvature(input);
  return 0.5 * weight_ * curvature * curvature;
}

void CurvatureCost::Quadraticize(const VectorXf& input, MatrixXf* hess,
                                 VectorXf* grad) const {
  CHECK_NOTNULL(hess);
  CHECK_NOTNULL(grad);

  // Check dimensions.
  CHECK_EQ(input.size(), hess->rows());
  CHECK_EQ(input.size(), hess->cols());
  CHECK_EQ(input.size(), grad->size());

  // Populate Hessian and gradient.
  const float v = input(v_idx_);
  const float omega = input(omega_idx_);
  const float one_over_vsq = 1.0 / (v * v);
  const float weight_over_vsq = weight_ * one_over_vsq;
  const float weight_omega_over_vsq = omega * weight_over_vsq;

  float domega = weight_omega_over_vsq;
  float dv = -weight_omega_over_vsq * omega / v;
  float ddomega = weight_over_vsq;
  float ddv = 3.0 * weight_omega_over_vsq * omega * one_over_vsq;
  float domega_dv = -2.0 * weight_omega_over_vsq / v;

  ModifyDerivatives(input, &domega, &ddomega, &dv, &ddv, &domega_dv);

  (*grad)(omega_idx_) += domega;
  (*grad)(v_idx_) += dv;

  (*hess)(omega_idx_, omega_idx_) += ddomega;
  (*hess)(omega_idx_, v_idx_) += domega_dv;
  (*hess)(v_idx_, omega_idx_) += domega_dv;
  (*hess)(v_idx_, v_idx_) += ddv;
}

}  // namespace ilqgames
