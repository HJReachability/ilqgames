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
// Quadratic generalized control cost for the flat 4D unicycle dynamics.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/cost/quadratic_control_cost_flat_unicycle_4d.h>
#include <ilqgames/dynamics/single_player_flat_unicycle_4d.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <string>

namespace ilqgames {

float QuadraticControlCostFlatUnicycle4D::Evaluate(const VectorXf& xi,
                                                   const VectorXf& v) const {
  CHECK_LT(dimension_, v.size());

  // Compute linearizing control for this subsystem.
  const auto& dyn =
      *static_cast<const ConcatenatedFlatSystem*>(dynamics_.get());
  const auto& subsystem = *dyn.Subsystems()[subsystem_idx_];
  const VectorXf u = subsystem.LinearizingControl(
      dyn.FromLinearSystemState(xi, subsystem_idx_), v);

  // If dimension non-negative, then just square the desired dimension.
  if (dimension_ >= 0) {
    const float delta = u(dimension_) - nominal_;
    return 0.5 * weight_ * delta * delta;
  }

  // Otherwise, cost is squared 2-norm of entire input.
  return 0.5 * weight_ *
         (u - VectorXf::Constant(u.size(), nominal_)).squaredNorm();
}

void QuadraticControlCostFlatUnicycle4D::Quadraticize(const VectorXf& xi,
                                                      const VectorXf& v,
                                                      MatrixXf* hess_v,
                                                      MatrixXf* hess_xi,
                                                      VectorXf* grad_xi) const {
  CHECK_LT(dimension_, v.size());
  CHECK_NOTNULL(hess_v);
  CHECK_NOTNULL(hess_xi);
  CHECK_NOTNULL(grad_xi);

  // Compute linearizing control for this subsystem.
  const auto& dyn =
      *static_cast<const ConcatenatedFlatSystem*>(dynamics_.get());
  const auto& subsystem = *dyn.Subsystems()[subsystem_idx_];
  const VectorXf x = dyn.FromLinearSystemState(xi, subsystem_idx_);
  const VectorXf u = subsystem.LinearizingControl(x, v);
  const Dimension start_dim = dyn.SubsystemStartDim(subsystem_idx_);

  // Check dimensions.
  CHECK_EQ(v.size(), hess_v->rows());
  CHECK_EQ(v.size(), hess_v->cols());
  CHECK_EQ(dyn.XDim(), hess_xi->rows());
  CHECK_EQ(dyn.XDim(), hess_xi->cols());
  CHECK_EQ(dyn.XDim(), grad_xi->size());

  // Unpack terms.
  const float v1 = v(0);
  const float v2 = v(1);
  const float v12 = v1 * v1;
  const float v22 = v2 * v2;

  const float vx = xi(start_dim + SinglePlayerFlatUnicycle4D::kVxIdx);
  const float vy = xi(start_dim + SinglePlayerFlatUnicycle4D::kVyIdx);
  const float vx2 = vx * vx;
  const float vx4 = vx2 * vx2;
  const float vy2 = vy * vy;
  const float vy4 = vy2 * vy2;

  const float s = x(SinglePlayerFlatUnicycle4D::kVIdx);
  const float s2 = s * s;
  const float s3 = s2 * s;
  const float s4 = s3 * s;
  const float s6 = s4 * s2;
  const float s8 = s4 * s4;

  // Populate gradient and Hessian terms below.
  float ddvx = 0.0;
  float ddvy = 0.0;
  float d2dvx2 = 0.0;
  float d2dvy2 = 0.0;
  float d2dvxdvy = 0.0;
  float d2dv12 = 0.0;
  float d2dv22 = 0.0;
  float d2dv1dv2 = 0.0;

  // Handle single dimension case first. Can either be an acceleration cost or a
  // omega cost.
  if (dimension_ == SinglePlayerFlatUnicycle4D::kOmegaIdx) {
    // State gradient.
    ddvx = -(2 * vy *
             (-v12 * vx * vy + v1 * v2 * vx2 - v1 * v2 * vy2 + v22 * vx * vy)) /
           (s4);
    ddvy = (2 * vx *
            (-v12 * vx * vy + v1 * v2 * vx2 - v1 * v2 * vy2 + v22 * vx * vy)) /
           (s4);

    d2dvx2 = ((8 * v12 - 8 * v22) * vy4 - 16 * v1 * v2 * vx * vy * vy2) / (s6) -
             ((6 * v12 - 6 * v22) * vy2 - 4 * v1 * v2 * vx * vy) / (s4);
    d2dvxdvy = (2 * (v1 * vx2 + 2 * v2 * vx * vy - v1 * vy2) *
                (-v2 * vx2 + 2 * v1 * vx * vy + v2 * vy2)) /
               (s6);
    d2dvy2 = ((6 * v12 - 6 * v22) * vx2 + 4 * v1 * v2 * vy * vx) / (s4) -
             ((8 * v12 - 8 * v22) * vx4 + 16 * v1 * v2 * vy * vx * vx2) / (s6);

    d2dv12 = (2 * vx2) / (s2);
    d2dv1dv2 = (2 * vx * vy) / (s2);
    d2dv22 = (2 * vy2) / (s2);
  } else if (dimension_ == SinglePlayerFlatUnicycle4D::kAIdx) {
    ddvx = -(4 * v12 * vx * vy2 - 6 * v1 * v2 * vx2 * vy +
             2 * v1 * v2 * vy * vy2 + 2 * v22 * vx * vx2 - 2 * v22 * vx * vy2) /
           (s6);
    ddvy = -(-2 * v12 * vx2 * vy + 2 * v12 * vy * vy2 + 2 * v1 * v2 * vx * vx2 -
             6 * v1 * v2 * vx * vy2 + 4 * v22 * vx2 * vy) /
           (s6);

    d2dvx2 = (6 * v22 * vx4 - vy4 * (4 * v12 - 2 * v22) +
              vx2 * vy2 * (20 * v12 - 16 * v22) + 24 * v1 * v2 * vx * vy * vy2 -
              24 * v1 * v2 * vx * vx2 * vy) /
             (s8);
    d2dvxdvy = (6 * v1 * v2 * vx4 + (16 * v22 - 8 * v12) * vx * vx2 * vy -
                36 * v1 * v2 * vx2 * vy2 +
                (16 * v12 - 8 * v22) * vx * vy * vy2 + 6 * v1 * v2 * vy4) /
               (s8);
    d2dvy2 = (6 * v12 * vy4 + vx4 * (2 * v12 - 4 * v22) -
              vx2 * vy2 * (16 * v12 - 20 * v22) - 24 * v1 * v2 * vx * vy * vy2 +
              24 * v1 * v2 * vx * vx2 * vy) /
             (s8);

    d2dv12 = (2 * vy2) / (s4);
    d2dv1dv2 = -(2 * vx * vy) / (s4);
    d2dv22 = (2 * vx2) / (s4);
  }
  // Handle isotropic case.
  else {
    // State gradient.
    ddvx = 2 * ((v2 * vx) / (s2) - (v1 * vy) / (s2)) *
               (v2 / (s2) - (2 * v2 * vx2) / (s4) + (2 * v1 * vx * vy) / (s4)) -
           2 * ((v1 * vx) / (s) + (v2 * vy) / (s)) *
               ((v1 * vx2) / (s3)-v1 / (s) + (v2 * vx * vy) / (s3));

    ddvy = -2 * ((v2 * vx) / (s2) - (v1 * vy) / (s2)) *
               (v1 / (s2) - (2 * v1 * vy2) / (s4) + (2 * v2 * vx * vy) / (s4)) -
           2 * ((v1 * vx) / (s) + (v2 * vy) / (s)) *
               ((v2 * vy2) / (s3)-v2 / (s) + (v1 * vx * vy) / (s3));

    // State Hessian.
    d2dvx2 =
        (vy4 * (8 * v12 - 8 * v22) -
         vx * (16 * v1 * v2 * vy * vy2 + 24 * v1 * v2 * vy) +
         vy2 * (20 * v12 - 28 * v22)) /
            (s6) +
        (6 * v22 - vy2 * (6 * v12 - 6 * v22) + 4 * v1 * v2 * vx * vy) / (s4) +
        (24 * vy * vy2 * (-vy * v12 + 2 * vx * v1 * v2 + vy * v22)) / (s8);

    d2dvxdvy = (vy4 * (10 * v1 * v2 * vx2 + 6 * v1 * v2) +
                vy * ((4 * v12 - 4 * v22) * vx * vx4 +
                      (16 * v22 - 8 * v12) * vx * vx2) -
                vy2 * (-10 * v1 * v2 * vx4 + 36 * v1 * v2 * vx2) +
                6 * v1 * v2 * vx4 - 2 * v1 * v2 * vx2 * vx4 -
                2 * v1 * v2 * vy2 * vy4 - vx * vy * vy4 * (4 * v12 - 4 * v22) +
                vx * vy * vy2 * (16 * v12 - 8 * v22)) /
               (s8);

    d2dvy2 =
        (6 * v12 + vx2 * (6 * v12 - 6 * v22) + 4 * v1 * v2 * vx * vy) / (s4) -
        (vy * (16 * v1 * v2 * vx * vx2 + 24 * v1 * v2 * vx) +
         vx4 * (8 * v12 - 8 * v22) + vx2 * (28 * v12 - 20 * v22)) /
            (s6) +
        (24 * vx * vx2 * (vx * v12 + 2 * vy * v1 * v2 - vx * v22)) / (s8);

    // Control Hessian.
    d2dv12 = (2 * vy2 * (vx2 + 1) + 2 * vx4) / (s4);
    d2dv1dv2 = (2 * vx * vy * (s2 - 1)) / (s4);
    d2dv22 = (2 * vx2 * (vy2 + 1) + 2 * vy4) / (s4);
  }

  // Populate grads and Hessians.
  (*grad_xi)(start_dim + SinglePlayerFlatUnicycle4D::kVxIdx) += ddvx;
  (*grad_xi)(start_dim + SinglePlayerFlatUnicycle4D::kVyIdx) += ddvy;

  (*hess_xi)(start_dim + SinglePlayerFlatUnicycle4D::kVxIdx,
             start_dim + SinglePlayerFlatUnicycle4D::kVxIdx) += d2dvx2;
  (*hess_xi)(start_dim + SinglePlayerFlatUnicycle4D::kVxIdx,
             start_dim + SinglePlayerFlatUnicycle4D::kVyIdx) += d2dvxdvy;
  (*hess_xi)(start_dim + SinglePlayerFlatUnicycle4D::kVyIdx,
             start_dim + SinglePlayerFlatUnicycle4D::kVxIdx) += d2dvxdvy;
  (*hess_xi)(start_dim + SinglePlayerFlatUnicycle4D::kVyIdx,
             start_dim + SinglePlayerFlatUnicycle4D::kVyIdx) += d2dvy2;

  (*hess_v)(0, 0) += d2dv12;
  (*hess_v)(0, 1) += d2dv1dv2;
  (*hess_v)(1, 0) += d2dv1dv2;
  (*hess_v)(1, 1) += d2dv22;
}

}  // namespace ilqgames
