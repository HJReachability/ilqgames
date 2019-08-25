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
// Single player dynamics modeling a car. 5 states and 2 control inputs.
// State is [x, y, theta, phi, v], control is [omega, a], and dynamics are:
//                     \dot px    = v cos theta
//                     \dot py    = v sin theta
//                     \dot theta = (v / L) * tan phi
//                     \dot phi   = omega
//                     \dot v     = a
// Please refer to
// https://www.sciencedirect.com/science/article/pii/S2405896316301215
// for further details.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/dynamics/single_player_flat_car_6d.h>
#include <ilqgames/dynamics/single_player_flat_system.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>

namespace ilqgames {

// Constexprs for state indices.
const Dimension SinglePlayerFlatCar6D::kNumXDims = 6;
const Dimension SinglePlayerFlatCar6D::kPxIdx = 0;
const Dimension SinglePlayerFlatCar6D::kPyIdx = 1;
const Dimension SinglePlayerFlatCar6D::kThetaIdx = 2;
const Dimension SinglePlayerFlatCar6D::kPhiIdx = 3;
const Dimension SinglePlayerFlatCar6D::kVIdx = 4;
const Dimension SinglePlayerFlatCar6D::kAIdx = 5;
const Dimension SinglePlayerFlatCar6D::kVxIdx = 2;
const Dimension SinglePlayerFlatCar6D::kVyIdx = 3;
const Dimension SinglePlayerFlatCar6D::kAxIdx = 4;
const Dimension SinglePlayerFlatCar6D::kAyIdx = 5;

// Constexprs for control indices.
const Dimension SinglePlayerFlatCar6D::kNumUDims = 2;
const Dimension SinglePlayerFlatCar6D::kOmegaIdx = 0;
const Dimension SinglePlayerFlatCar6D::kJerkIdx = 1;

void SinglePlayerFlatCar6D::Partial(const VectorXf& xi,
                                    std::vector<VectorXf>* grads,
                                    std::vector<MatrixXf>* hesses) const {
  CHECK_NOTNULL(grads);
  CHECK_NOTNULL(hesses);

  // grads->clear();
  // grads->resize(xi.size(), VectorXf::Zero(kNumXDims));

  // hesses->clear();
  // hesses->resize(xi.size(), MatrixXf::Zero(kNumXDims, kNumXDims));

  if (grads->size() != xi.size())
    grads->resize(xi.size(), VectorXf::Zero(kNumXDims));
  else {
    for (auto& grad : *grads) {
      DCHECK_EQ(grad.size(), xi.size());
      grad.setZero();
    }
  }

  if (hesses->size() != xi.size())
    hesses->resize(xi.size(), MatrixXf::Zero(kNumXDims, kNumXDims));
  else {
    for (auto& hess : *hesses) {
      DCHECK_EQ(hess.rows(), xi.size());
      DCHECK_EQ(hess.cols(), xi.size());
      hess.setZero();
    }
  }

  const float vx = xi(kVxIdx);
  const float vy = xi(kVyIdx);
  const float ax = xi(kAxIdx);
  const float ay = xi(kAyIdx);
  const float L = inter_axle_distance_;

  CHECK_GT(std::hypot(vx, vy), 1e-2);

  const float vx2 = vx * vx;
  const float vx3 = vx2 * vx;
  const float vx4 = vx3 * vx;
  const float vx5 = vx4 * vx;
  const float vx6 = vx5 * vx;
  const float vx7 = vx6 * vx;
  const float vx8 = vx7 * vx;
  const float vx9 = vx8 * vx;

  const float vy2 = vy * vy;
  const float vy3 = vy2 * vy;
  const float vy4 = vy3 * vy;
  const float vy5 = vy4 * vy;
  const float vy6 = vy5 * vy;
  const float vy7 = vy6 * vy;
  const float vy8 = vy7 * vy;
  const float vy9 = vy8 * vy;

  const float ax2 = ax * ax;
  const float ax3 = ax2 * ax;
  const float ay2 = ay * ay;
  const float ay3 = ay2 * ay;

  const float L2 = L * L;

  const float norm_squared = vx2 + vy2;
  const float norm = std::sqrt(norm_squared);
  const float norm_ss = norm_squared * norm_squared;
  const float norm_sss = norm_ss * norm_squared;
  const float sqrt_norm_sss = std::sqrt(norm_sss);

  (*grads)[kPxIdx](kPxIdx) = 1.0;
  (*grads)[kPyIdx](kPyIdx) = 1.0;
  (*grads)[kThetaIdx](kVxIdx) = -vy / norm_squared;
  (*grads)[kThetaIdx](kVyIdx) = vx / norm_squared;
  (*grads)[kVIdx](kVxIdx) = vx / norm;
  (*grads)[kVIdx](kVyIdx) = vy / norm;

  (*grads)[kAIdx](kVxIdx) = vy * (ax * vy - ay * vx) / sqrt_norm_sss;
  (*grads)[kAIdx](kVyIdx) = vx * (ay * vx - ax * vy) / sqrt_norm_sss;
  (*grads)[kAIdx](kAxIdx) = (*grads)[kVIdx](kVxIdx);
  (*grads)[kAIdx](kAyIdx) = (*grads)[kVIdx](kVyIdx);
  (*grads)[kPhiIdx](kVxIdx) =
      (L * norm * (-2.0 * ay * vx2 + 3.0 * ax * vx * vy + ay * vy2)) /
      (L * L * ax2 * vy2 - 2.0 * L2 * ax * ay * vx * vy + L2 * ay2 * vx2 + vx6 +
       3.0 * vx4 * vy2 + 3.0 * vx2 * vy4 + vy4);
  (*grads)[kPhiIdx](kVyIdx) =
      -(L * norm * (ax * vx2 + 3.0 * ay * vx * vy - 2.0 * ax * vy2)) /
      (L2 * ax * ax * vy2 - 2.0 * L2 * ax * ay * vx * vy + L2 * ay2 * vx2 +
       vx6 + 3.0 * vx4 * vy2 + 3.0 * vx2 * vy4 + vy6);
  (*grads)[kPhiIdx](kAxIdx) = -(L * vy * sqrt_norm_sss) /
                              (norm_ss * norm_squared + L2 * ax2 * vy2 +
                               L2 * ay2 * vx2 - 2.0 * L2 * ax * ay * vx * vy);
  (*grads)[kPhiIdx](kAyIdx) = (L * vx * sqrt_norm_sss) /
                              (norm_ss * norm_squared + L2 * ax2 * vy2 +
                               L2 * ay2 * vx2 - 2.0 * L2 * ax * ay * vx * vy);

  (*hesses)[kThetaIdx](kVxIdx, kVxIdx) = 2.0 * vx * vy / norm_ss;
  (*hesses)[kThetaIdx](kVxIdx, kVyIdx) = (vy * vy - vx * vx) / norm_ss;
  (*hesses)[kThetaIdx](kVyIdx, kVxIdx) = (*hesses)[kThetaIdx](kVxIdx, kVyIdx);
  (*hesses)[kThetaIdx](kVyIdx, kVyIdx) = -(*hesses)[kThetaIdx](kVxIdx, kVxIdx);
  (*hesses)[kVIdx](kVxIdx, kVxIdx) = (vy * vy) / sqrt_norm_sss;
  (*hesses)[kVIdx](kVxIdx, kVyIdx) = (-vx * vy) / sqrt_norm_sss;
  (*hesses)[kVIdx](kVyIdx, kVxIdx) = (*hesses)[kVIdx](kVxIdx, kVyIdx);
  (*hesses)[kVIdx](kVyIdx, kVyIdx) = (vx * vx) / sqrt_norm_sss;

  (*hesses)[kAIdx](kVxIdx, kVxIdx) =
      -(vy * (-2.0 * ay * vx2 + 3.0 * ax * vx * vy + ay * vy2)) /
      (sqrt_norm_sss * norm);
  (*hesses)[kAIdx](kVxIdx, kVyIdx) =
      -(ay * vx3 - 2.0 * ax * vx2 * vy - 2.0 * ay * vx * vy2 + ax * vy3) /
      (sqrt_norm_sss * norm);
  (*hesses)[kAIdx](kVxIdx, kAxIdx) = vy2 / sqrt_norm_sss;
  (*hesses)[kAIdx](kVxIdx, kAyIdx) = -(vx * vy) / sqrt_norm_sss;
  (*hesses)[kAIdx](kVyIdx, kVxIdx) = (*hesses)[kAIdx](kVxIdx, kVyIdx);
  (*hesses)[kAIdx](kVyIdx, kVyIdx) =
      -(vx * (ax * vx2 + 3.0 * ay * vx * vy - 2.0 * ax * vy2)) /
      (sqrt_norm_sss * norm);
  (*hesses)[kAIdx](kVyIdx, kAxIdx) = -(vx * vy) / sqrt_norm_sss;
  (*hesses)[kAIdx](kVyIdx, kAyIdx) = vx2 / sqrt_norm_sss;
  (*hesses)[kAIdx](kAxIdx, kVxIdx) = (*hesses)[kAIdx](kVxIdx, kAxIdx);
  (*hesses)[kAIdx](kAxIdx, kVyIdx) = (*hesses)[kAIdx](kVyIdx, kAxIdx);
  (*hesses)[kAIdx](kAyIdx, kVxIdx) = (*hesses)[kAIdx](kVxIdx, kAyIdx);
  (*hesses)[kAIdx](kAyIdx, kVyIdx) = (*hesses)[kAIdx](kVyIdx, kAyIdx);

  const float denom =
      (L2 * ax2 * vy2 - 2 * L2 * ax * ay * vx * vy + L2 * ay2 * vx2 + vx6 +
       3 * vx4 * vy2 + 3 * vx2 * vy4 + vy6);
  const float denom2 =
      (norm_sss + L2 * ax2 * vy2 + L2 * ay2 * vx2 - 2 * L2 * ax * ay * vx * vy);
  (*hesses)[kPhiIdx](kVxIdx, kVxIdx) =
      -(L *
        (-6 * L2 * ax3 * vx2 * vy3 - 3 * L2 * ax3 * vy5 +
         12 * L2 * ax2 * ay * vx3 * vy2 + 3 * L2 * ax2 * ay * vx * vy4 -
         8 * L2 * ax * ay2 * vx4 * vy - L2 * ax * ay2 * vx2 * vy3 -
         2 * L2 * ax * ay2 * vy5 + 2 * L2 * ay3 * vx5 + L2 * ay3 * vx3 * vy2 +
         2 * L2 * ay3 * vx * vy4 + 12 * ax * vx8 * vy + 33 * ax * vx6 * vy3 +
         27 * ax * vx4 * vy5 + 3 * ax * vx2 * vy7 - 3 * ax * vy9 -
         6 * ay * vx9 - 9 * ay * vx7 * vy2 + 9 * ay * vx5 * vy4 +
         21 * ay * vx3 * vy6 + 9 * ay * vx * vy8)) /
      (norm * denom * denom);
  (*hesses)[kPhiIdx](kVxIdx, kVyIdx) =
      (L *
       (-3 * L2 * ax3 * vx3 * vy2 + 4 * L2 * ax2 * ay * vx4 * vy -
        4 * L2 * ax2 * ay * vx2 * vy3 + L2 * ax2 * ay * vy5 -
        L2 * ax * ay2 * vx5 + 4 * L2 * ax * ay2 * vx3 * vy2 -
        4 * L2 * ax * ay2 * vx * vy4 + 3 * L2 * ay3 * vx2 * vy3 + 3 * ax * vx9 -
        3 * ax * vx7 * vy2 - 27 * ax * vx5 * vy4 - 33 * ax * vx3 * vy6 -
        12 * ax * vx * vy8 + 12 * ay * vx8 * vy + 33 * ay * vx6 * vy3 +
        27 * ay * vx4 * vy5 + 3 * ay * vx2 * vy7 - 3 * ay * vy9)) /
      (norm * denom * denom);
  (*hesses)[kPhiIdx](kVxIdx, kAxIdx) =
      (L * vy * norm *
       (-3 * L2 * ax2 * vx * vy2 + 4 * L2 * ax * ay * vx2 * vy -
        2 * L2 * ax * ay * vy3 - L2 * ay2 * vx3 + 2 * L2 * ay2 * vx * vy2 +
        3 * vx7 + 9 * vx5 * vy2 + 9 * vx3 * vy4 + 3 * vx * vy6)) /
      (denom * denom);
  (*hesses)[kPhiIdx](kVxIdx, kAyIdx) =
      (L * norm *
       (4 * L2 * ax2 * vx2 * vy2 + L2 * ax2 * vy4 -
        6 * L2 * ax * ay * vx3 * vy + 2 * L2 * ay2 * vx4 -
        L2 * ay2 * vx2 * vy2 - 2 * vx8 - 5 * vx6 * vy2 - 3 * vx4 * vy4 +
        vx2 * vy6 + vy8)) /
      (denom * denom);
  (*hesses)[kPhiIdx](kVyIdx, kVxIdx) = (*hesses)[kAIdx](kVxIdx, kVyIdx);
  (*hesses)[kPhiIdx](kVyIdx, kVyIdx) =
      (L * (2 * L2 * ax3 * vx4 * vy + L2 * ax3 * vx2 * vy3 +
            2 * L2 * ax3 * vy5 - 2 * L2 * ax2 * ay * vx5 -
            L2 * ax2 * ay * vx3 * vy2 - 8 * L2 * ax2 * ay * vx * vy4 +
            3 * L2 * ax * ay2 * vx4 * vy + 12 * L2 * ax * ay2 * vx2 * vy3 -
            3 * L2 * ay3 * vx5 - 6 * L2 * ay3 * vx3 * vy2 + 9 * ax * vx8 * vy +
            21 * ax * vx6 * vy3 + 9 * ax * vx4 * vy5 - 9 * ax * vx2 * vy7 -
            6 * ax * vy9 - 3 * ay * vx9 + 3 * ay * vx7 * vy2 +
            27 * ay * vx5 * vy4 + 33 * ay * vx3 * vy6 + 12 * ay * vx * vy8)) /
      (norm * denom * denom);
  (*hesses)[kPhiIdx](kVyIdx, kAxIdx) =
      (L * norm *
       (L2 * ax2 * vx2 * vy2 - 2 * L2 * ax2 * vy4 +
        6 * L2 * ax * ay * vx * vy3 - L2 * ay2 * vx4 -
        4 * L2 * ay2 * vx2 * vy2 - vx8 - vx6 * vy2 + 3 * vx4 * vy4 +
        5 * vx2 * vy6 + 2 * vy8)) /
      (denom * denom);
  (*hesses)[kPhiIdx](kVyIdx, kAyIdx) =
      -(L * vx * norm *
        (2 * L2 * ax2 * vx2 * vy - L2 * ax2 * vy3 - 2 * L2 * ax * ay * vx3 +
         4 * L2 * ax * ay * vx * vy2 - 3 * L2 * ay2 * vx2 * vy + 3 * vx6 * vy +
         9 * vx4 * vy3 + 9 * vx2 * vy5 + 3 * vy7)) /
      (denom * denom);
  (*hesses)[kPhiIdx](kAxIdx, kVxIdx) = (*hesses)[kPhiIdx](kVxIdx, kAxIdx);
  (*hesses)[kPhiIdx](kAxIdx, kVyIdx) = (*hesses)[kPhiIdx](kVyIdx, kAxIdx);
  (*hesses)[kPhiIdx](kAxIdx, kAxIdx) =
      (L * vy * sqrt_norm_sss * (2 * ax * L2 * vy2 - 2 * ay * vx * L2 * vy)) /
      (denom2 * denom2);
  (*hesses)[kPhiIdx](kAxIdx, kAyIdx) =
      (L * vy * sqrt_norm_sss * (2 * ay * L2 * vx2 - 2 * ax * vy * L2 * vx)) /
      (denom2 * denom2);
  (*hesses)[kPhiIdx](kAyIdx, kVxIdx) = (*hesses)[kPhiIdx](kVxIdx, kAyIdx);
  (*hesses)[kPhiIdx](kAyIdx, kVyIdx) = (*hesses)[kPhiIdx](kVyIdx, kAyIdx);
  (*hesses)[kPhiIdx](kAyIdx, kAxIdx) = (*hesses)[kPhiIdx](kAxIdx, kAyIdx);
  (*hesses)[kPhiIdx](kAyIdx, kAyIdx) =
      -(L * vx * sqrt_norm_sss * (2 * ay * L2 * vx2 - 2 * ax * vy * L2 * vx)) /
      (denom2 * denom2);
}

}  // namespace ilqgames
