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
// State is [x, y, theta, phi, v, a], control is [omega, j], and dynamics are:
//                     \dot px    = v cos theta
//                     \dot py    = v sin theta
//                     \dot theta = (v / L) * tan phi
//                     \dot phi   = omega
//                     \dot v     = a
//                     \dot a     = j
// Please refer to
// https://www.sciencedirect.com/science/article/pii/S2405896316301215
// for further details.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_DYNAMICS_SINGLE_PLAYER_FLAT_CAR_6D_H
#define ILQGAMES_DYNAMICS_SINGLE_PLAYER_FLAT_CAR_6D_H

#include <ilqgames/dynamics/single_player_flat_system.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>

namespace ilqgames {

class SinglePlayerFlatCar6D : public SinglePlayerFlatSystem {
 public:
  ~SinglePlayerFlatCar6D() {}
  SinglePlayerFlatCar6D(float inter_axle_distance)
      : SinglePlayerFlatSystem(kNumXDims, kNumUDims),
        inter_axle_distance_(inter_axle_distance) {}

  // Compute time derivative of state.
  VectorXf Evaluate(const VectorXf& x, const VectorXf& u) const;

  // Discrete time approximation of the underlying linearized system.
  void LinearizedSystem(Eigen::Ref<MatrixXf> A, Eigen::Ref<MatrixXf> B) const;

  // Utilities for feedback linearization.
  MatrixXf InverseDecouplingMatrix(const VectorXf& x) const;
  VectorXf AffineTerm(const VectorXf& x) const;
  VectorXf ToLinearSystemState(const VectorXf& x) const;
  VectorXf FromLinearSystemState(const VectorXf& xi) const;
  void Partial(const VectorXf& xi, std::vector<VectorXf>* grads,
               std::vector<MatrixXf>* hesses) const;
  bool IsLinearSystemStateSingular(const VectorXf& xi) const;

  // Distance metric between two states.
  float DistanceBetween(const VectorXf& x0, const VectorXf& x1) const;

  // Position dimensions.
  std::vector<Dimension> PositionDimensions() const { return {kPxIdx, kPyIdx}; }

  // Constexprs for state indices.
  static const Dimension kNumXDims;
  static const Dimension kPxIdx;
  static const Dimension kPyIdx;
  static const Dimension kThetaIdx;
  static const Dimension kPhiIdx;
  static const Dimension kVIdx;
  static const Dimension kAIdx;
  static const Dimension kVxIdx;
  static const Dimension kVyIdx;
  static const Dimension kAxIdx;
  static const Dimension kAyIdx;

  // Constexprs for control indices.
  static const Dimension kNumUDims;
  static const Dimension kOmegaIdx;
  static const Dimension kJerkIdx;

 private:
  // Inter-axle distance. Determines turning radius.
  const float inter_axle_distance_;
};  //\class SinglePlayerCar6D

// ----------------------------- IMPLEMENTATION ----------------------------- //

inline VectorXf SinglePlayerFlatCar6D::Evaluate(const VectorXf& x,
                                                const VectorXf& u) const {
  VectorXf xdot(xdim_);
  xdot(kPxIdx) = x(kVIdx) * std::cos(x(kThetaIdx));
  xdot(kPyIdx) = x(kVIdx) * std::sin(x(kThetaIdx));
  xdot(kThetaIdx) = (x(kVIdx) / inter_axle_distance_) * std::tan(x(kPhiIdx));
  xdot(kPhiIdx) = u(kOmegaIdx);
  xdot(kVIdx) = x(kAIdx);
  xdot(kAIdx) = u(kJerkIdx);

  return xdot;
}

inline void SinglePlayerFlatCar6D::LinearizedSystem(
    Eigen::Ref<MatrixXf> A, Eigen::Ref<MatrixXf> B) const {
  A(kPxIdx, kVxIdx) += time::kTimeStep;
  A(kPyIdx, kVyIdx) += time::kTimeStep;
  A(kVxIdx, kAxIdx) += time::kTimeStep;
  A(kVyIdx, kAyIdx) += time::kTimeStep;

  B(kAxIdx, 0) = time::kTimeStep;
  B(kAyIdx, 1) = time::kTimeStep;
}

inline MatrixXf SinglePlayerFlatCar6D::InverseDecouplingMatrix(
    const VectorXf& x) const {
  MatrixXf M_inv(kNumUDims, kNumUDims);

  const float sin_t = std::sin(x(kThetaIdx));
  const float cos_t = std::cos(x(kThetaIdx));
  // HACK! KSmallOffset should realy be 0...
  const float kSmallOffset = sgn(x(kVIdx) + 0.0000001) * 0.00011;
  const float cos_phi_v = std::cos(x(kPhiIdx)) / (x(kVIdx) + kSmallOffset);
  const float scaling = inter_axle_distance_ * cos_phi_v * cos_phi_v;

  CHECK_GT(std::abs(x(kVIdx) + kSmallOffset), 1e-4);

  M_inv(0, 0) = -scaling * sin_t;
  M_inv(0, 1) = scaling * cos_t;
  M_inv(1, 0) = cos_t;
  M_inv(1, 1) = sin_t;

  return M_inv;
}

inline VectorXf SinglePlayerFlatCar6D::AffineTerm(const VectorXf& x) const {
  VectorXf m = VectorXf::Zero(kNumUDims);

  const float sin_t = std::sin(x(kThetaIdx));
  const float cos_t = std::cos(x(kThetaIdx));
  const float tan_phi = std::tan(x(kPhiIdx));
  const float v_over_l = x(kVIdx) / inter_axle_distance_;

  m(0) = -v_over_l * tan_phi *
         (3.0 * x(kAIdx) * sin_t + v_over_l * x(kVIdx) * tan_phi * cos_t);
  m(1) = v_over_l * tan_phi *
         (3.0 * x(kAIdx) * cos_t - v_over_l * x(kVIdx) * tan_phi * sin_t);

  return m;
}

inline VectorXf SinglePlayerFlatCar6D::ToLinearSystemState(
    const VectorXf& x) const {
  VectorXf xi(kNumXDims);

  const float sin_t = std::sin(x(kThetaIdx));
  const float cos_t = std::cos(x(kThetaIdx));
  const float tan_phi = std::tan(x(kPhiIdx));
  const float vv_over_l = x(kVIdx) * x(kVIdx) / inter_axle_distance_;

  xi(kPxIdx) = x(kPxIdx);
  xi(kPyIdx) = x(kPyIdx);
  xi(kVxIdx) = x(kVIdx) * cos_t;
  xi(kVyIdx) = x(kVIdx) * sin_t;
  xi(kAxIdx) = x(kAIdx) * cos_t - vv_over_l * sin_t * tan_phi;
  xi(kAyIdx) = x(kAIdx) * sin_t + vv_over_l * cos_t * tan_phi;

  return xi;
}

inline VectorXf SinglePlayerFlatCar6D::FromLinearSystemState(
    const VectorXf& xi) const {
  VectorXf x(kNumXDims);

  x(kPxIdx) = xi(kPxIdx);
  x(kPyIdx) = xi(kPyIdx);
  x(kThetaIdx) = std::atan2(xi(kVyIdx), xi(kVxIdx));
  x(kVIdx) = std::hypot(xi(kVyIdx), xi(kVxIdx));

  const float cos_t = xi(kVxIdx) / x(kVIdx);
  const float sin_t = xi(kVyIdx) / x(kVIdx);

  x(kAIdx) = cos_t * xi(kAxIdx) + sin_t * xi(kAyIdx);
  x(kPhiIdx) = std::atan((x(kAIdx) * cos_t - xi(kAxIdx)) *
                         inter_axle_distance_ / (x(kVIdx) * x(kVIdx) * sin_t));

  return x;
}

inline bool SinglePlayerFlatCar6D::IsLinearSystemStateSingular(
    const VectorXf& xi) const {
  constexpr float kTolerance = 1e-2;
  return (std::isnan(xi(kVxIdx)) || std::isnan(xi(kVyIdx))) ||
         (std::abs(xi(kVxIdx)) < kTolerance &&
          std::abs(xi(kVyIdx)) < kTolerance);
}

inline float SinglePlayerFlatCar6D::DistanceBetween(const VectorXf& x0,
                                                    const VectorXf& x1) const {
  // Squared distance in position space.
  const float dx = x0(kPxIdx) - x1(kPxIdx);
  const float dy = x0(kPyIdx) - x1(kPyIdx);
  return dx * dx + dy * dy;
}

}  // namespace ilqgames

#endif
