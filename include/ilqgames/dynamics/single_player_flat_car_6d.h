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

#ifndef ILQGAMES_DYNAMICS_SINGLE_PLAYER_FLAT_CAR_6D_H
#define ILQGAMES_DYNAMICS_SINGLE_PLAYER_FLAT_CAR_6D_H

#include <ilqgames/dynamics/single_player_flat_system.h>
#include <ilqgames/utils/types.h>

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
  void LinearizeSystem(Time time_step,
                 Eigen::Ref<MatrixXf> A, Eigen::Ref<MatrixXf> B) const;

  // Utilities for feedback linearization.
  MatrixXf InverseDecouplingMatrix(const VectorXf& x) const;

  VectorXf AffineTerm(const VectorXf& x) const;

  VectorXf ToLinearSystemState(const VectorXf& x) const;

  VectorXf FromLinearSystemState(const VectorXf& xi) const;

  void Partial(const VectorXf& xi, std::vector<VectorXf>* grads, 
              std::vector<MatrixXf>* hesses) const;

  // Constexprs for state indices.
  static constexpr Dimension kNumXDims = 6;
  static constexpr Dimension kPxIdx = 0;
  static constexpr Dimension kPyIdx = 1;
  static constexpr Dimension kThetaIdx = 2;
  static constexpr Dimension kPhiIdx = 3;
  static constexpr Dimension kVIdx = 4;
  static constexpr Dimension kAIdx = 5;
  static constexpr Dimension kVxIdx = 2;
  static constexpr Dimension kVyIdx = 3;
  static constexpr Dimension kAxIdx = 4;
  static constexpr Dimension kAyIdx = 5;

  // Constexprs for control indices.
  static constexpr Dimension kNumUDims = 2;
  static constexpr Dimension kOmegaIdx = 0;
  static constexpr Dimension kJerkIdx = 1;

 private:
  // Inter-axle distance. Determines turning radius.
  const float inter_axle_distance_;
};  //\class SinglePlayerCar5D

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
                                         Time time_step,
                                         Eigen::Ref<MatrixXf> A,
                                         Eigen::Ref<MatrixXf> B) const {

  A(kPxIdx, kVxIdx) += time_step;
  A(kPyIdx, kVyIdx) += time_step;
  A(kVxIdx, kAxIdx) += time_step;
  A(kVyIdx, kAyIdx) += time_step;

  B(kAxIdx, 0) = time_step;
  B(kAyIdx, 1) = time_step;
}

inline MatrixXf SinglePlayerFlatCar6D::InverseDecouplingMatrix(const VectorXf& x) const{
  MatrixXf M_inv(kNumUDims,kNumUDims);

  const float sin_t = std::sin(x(kThetaIdx));
  const float cos_t = std::cos(x(kThetaIdx));
  const float cos_phi_v = std::cos(x(kPhiIdx))/x(kVIdx);
  const float scaling = inter_axle_distance_ * cos_phi_v * cos_phi_v;
   
  M_inv(0,0) = -scaling * sin_t;
  M_inv(0,1) =  scaling * cos_t;
  M_inv(1,0) =  cos_t;
  M_inv(1,1) =  sin_t;

  return M_inv;
}

inline VectorXf SinglePlayerFlatCar6D::AffineTerm(const VectorXf& x) const{
  VectorXf m = VectorXf::Zero(kNumUDims);

  const float sin_t = std::sin(x(kThetaIdx));
  const float cos_t = std::cos(x(kThetaIdx));
  const float tan_phi = std::tan(x(kPhiIdx));
  const float v_over_l = x(kVIdx)/inter_axle_distance_;

  m(0) = -v_over_l * tan_phi * (3.0 * x(kAIdx) * sin_t + 
                                v_over_l * x(kVIdx) * tan_phi * cos_t);
  m(1) =  v_over_l * tan_phi * (3.0 * x(kAIdx) * cos_t - 
                                v_over_l * x(kVIdx) * tan_phi * sin_t); 

  return m;
}

inline VectorXf SinglePlayerFlatCar6D::ToLinearSystemState(const VectorXf& x) const{
  VectorXf xi(kNumXDims);

  const float sin_t = std::sin(x(kThetaIdx));
  const float cos_t = std::cos(x(kThetaIdx));
  const float tan_phi = std::tan(x(kPhiIdx));
  const float vv_over_l = x(kVIdx) * x(kVIdx)/inter_axle_distance_;

  xi(kPxIdx) = x(kPxIdx);
  xi(kPyIdx) = x(kPyIdx);
  xi(kVxIdx) = x(kVIdx) * cos_t;
  xi(kVyIdx) = x(kVIdx) * sin_t;
  xi(kAxIdx) = x(kAIdx) * cos_t - vv_over_l * sin_t * tan_phi;
  xi(kAyIdx) = x(kAIdx) * sin_t + vv_over_l * cos_t * tan_phi;

  return xi;
}

inline VectorXf SinglePlayerFlatCar6D::FromLinearSystemState(const VectorXf& xi) const{
  VectorXf x(kNumXDims);

  x(kPxIdx) = xi(kPxIdx);
  x(kPyIdx) = xi(kPyIdx);
  x(kThetaIdx) = std::atan2(xi(kVyIdx),xi(kVxIdx));
  x(kVIdx) = std::hypot(xi(kVyIdx),xi(kVxIdx));

  const float cos_t = xi(kVxIdx) / x(kVIdx);
  const float sin_t = xi(kVyIdx) / x(kVIdx);

  x(kAIdx) = cos_t * xi(kAxIdx) + sin_t * xi(kAyIdx);
  x(kPhiIdx) = std::atan((x(kAIdx) * cos_t - xi(kAxIdx)) * 
                  inter_axle_distance_ / (x(kVIdx) * x(kVIdx) * sin_t));

  return x;
}

inline void SinglePlayerFlatCar6D::Partial(const VectorXf& xi, 
              std::vector<VectorXf>* grads, std::vector<MatrixXf>* hesses) const {
  CHECK_NOTNULL(grads);
  CHECK_NOTNULL(hesses);

  grads->resize(xi.size(),VectorXf::Zero(kNumXDims));
  hesses->resize(xi.size(),MatrixXf::Zero(kNumXDims,kNumXDims));

  const float norm_squared = xi(kVxIdx) * xi(kVxIdx) + xi(kVyIdx) * xi(kVyIdx);
  const float norm = std::sqrt(norm_squared);
  const float norm_ss = norm_squared * norm_squared;
  const float sqrt_norm_sss = std::sqrt(norm_ss * norm_squared);

  (*grads)[kPxIdx](kPxIdx) = 1.0;
  (*grads)[kPyIdx](kPyIdx) = 1.0;
  (*grads)[kThetaIdx](kVxIdx) = -xi(kVyIdx)/norm_squared;
  (*grads)[kThetaIdx](kVyIdx) = xi(kVxIdx)/norm_squared;
  (*grads)[kVIdx](kVxIdx) = xi(kVxIdx)/norm;
  (*grads)[kVIdx](kVyIdx) = xi(kVyIdx)/norm;
  
  (*grads)[kAIdx](kVxIdx) = xi(kVyIdx)*(xi(kAxIdx) * xi(kVyIdx) - xi(kAyIdx) * xi(kVxIdx))/sqrt_norm_sss;
  (*grads)[kAIdx](kVyIdx) = xi(kVxIdx)*(xi(kAyIdx) * xi(kVxIdx) - xi(kAxIdx) * xi(kVyIdx))/sqrt_norm_sss;
  (*grads)[kAIdx](kAxIdx) = (*grads)[kVIdx](kVxIdx);
  (*grads)[kAIdx](kAyIdx) = (*grads)[kVIdx](kVyIdx);

  (*grads)[kPhiIdx](kVxIdx) = // TODO
  (*grads)[kPhiIdx](kVyIdx) = // TODO
  (*grads)[kPhiIdx](kAxIdx) = // TODO
  (*grads)[kPhiIdx](kAyIdx) = // TODO

  (*hesses)[kThetaIdx](kVxIdx, kVxIdx) = 2.0 * xi(kVxIdx) * xi(kVyIdx)/norm_ss;
  (*hesses)[kThetaIdx](kVxIdx, kVyIdx) = (xi(kVyIdx)*xi(kVyIdx) - xi(kVxIdx)*xi(kVxIdx))/norm_ss;
  (*hesses)[kThetaIdx](kVyIdx, kVxIdx) = hesses[kThetaIdx](kVxIdx, kVyIdx);
  (*hesses)[kThetaIdx](kVyIdx, kVyIdx) = -hesses[kThetaIdx](kVxIdx, kVxIdx);
  (*hesses)[kVIdx](kVxIdx, kVxIdx) = (xi(kVyIdx) * xi(kVyIdx))/sqrt_norm_sss;
  (*hesses)[kVIdx](kVxIdx, kVyIdx) = (-xi(kVxIdx) * xi(kVyIdx))/sqrt_norm_sss;
  (*hesses)[kVIdx](kVyIdx, kVxIdx) = hesses[kVIdx](kVxIdx, kVyIdx);
  (*hesses)[kVIdx](kVyIdx, kVyIdx) = (xi(kVxIdx) * xi(kVxIdx))/sqrt_norm_sss;

}

}  // namespace ilqgames

#endif
