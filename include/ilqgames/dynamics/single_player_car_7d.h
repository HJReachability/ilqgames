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
// Single player dynamics modeling a car. 6 states and 2 control inputs.
// State is [x, y, theta, phi, v, kappa, s], control is [omega, a], and dynamics
// are:
//                     \dot px    = v cos theta
//                     \dot py    = v sin theta
//                     \dot theta = (v / L) * tan phi
//                     \dot phi   = omega
//                     \dot v     = a
//                     \dot kappa = omega / (L * cos^2 phi)   [ curvature ]
//                     \dot s     = v                         [ path length ]
// Please refer to
// https://www.sciencedirect.com/science/article/pii/S2405896316301215
// for further details.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_DYNAMICS_SINGLE_PLAYER_CAR_7D_H
#define ILQGAMES_DYNAMICS_SINGLE_PLAYER_CAR_7D_H

#include <ilqgames/dynamics/single_player_dynamical_system.h>
#include <ilqgames/utils/types.h>

namespace ilqgames {

class SinglePlayerCar7D : public SinglePlayerDynamicalSystem {
 public:
  ~SinglePlayerCar7D() {}
  SinglePlayerCar7D(float inter_axle_distance)
      : SinglePlayerDynamicalSystem(kNumXDims, kNumUDims),
        inter_axle_distance_(inter_axle_distance) {}

  // Compute time derivative of state.
  VectorXf Evaluate(Time t, const VectorXf& x, const VectorXf& u) const;

  // Compute a discrete-time Jacobian linearization.
  void Linearize(Time t, Time time_step, const VectorXf& x, const VectorXf& u,
                 Eigen::Ref<MatrixXf> A, Eigen::Ref<MatrixXf> B) const;

  // Distance metric between two states.
  float DistanceBetween(const VectorXf& x0, const VectorXf& x1) const;

  // Constexprs for state indices.
  static constexpr Dimension kNumXDims = 7;
  static constexpr Dimension kPxIdx = 0;
  static constexpr Dimension kPyIdx = 1;
  static constexpr Dimension kThetaIdx = 2;
  static constexpr Dimension kPhiIdx = 3;
  static constexpr Dimension kVIdx = 4;
  static constexpr Dimension kKappaIdx = 5;
  static constexpr Dimension kSIdx = 6;

  // Constexprs for control indices.
  static constexpr Dimension kNumUDims = 2;
  static constexpr Dimension kOmegaIdx = 0;
  static constexpr Dimension kAIdx = 1;

 private:
  // Inter-axle distance. Determines turning radius.
  const float inter_axle_distance_;
};  //\class SinglePlayerCar7D

// ----------------------------- IMPLEMENTATION ----------------------------- //

inline VectorXf SinglePlayerCar7D::Evaluate(Time t, const VectorXf& x,
                                            const VectorXf& u) const {
  VectorXf xdot(xdim_);
  xdot(kPxIdx) = x(kVIdx) * std::cos(x(kThetaIdx));
  xdot(kPyIdx) = x(kVIdx) * std::sin(x(kThetaIdx));
  xdot(kThetaIdx) = (x(kVIdx) / inter_axle_distance_) * std::tan(x(kPhiIdx));
  xdot(kPhiIdx) = u(kOmegaIdx);
  xdot(kVIdx) = u(kAIdx);

  const float sec_phi = 1.0 / std::cos(x(kPhiIdx));
  xdot(kKappaIdx) = u(kOmegaIdx) * sec_phi * sec_phi / inter_axle_distance_;
  xdot(kSIdx) = x(kVIdx);

  return xdot;
}

inline void SinglePlayerCar7D::Linearize(Time t, Time time_step,
                                         const VectorXf& x, const VectorXf& u,
                                         Eigen::Ref<MatrixXf> A,
                                         Eigen::Ref<MatrixXf> B) const {
  const float ctheta = std::cos(x(kThetaIdx)) * time_step;
  const float stheta = std::sin(x(kThetaIdx)) * time_step;
  const float cphi = std::cos(x(kPhiIdx));
  const float tphi = std::tan(x(kPhiIdx));

  A(kPxIdx, kThetaIdx) += -x(kVIdx) * stheta;
  A(kPxIdx, kVIdx) += ctheta;

  A(kPyIdx, kThetaIdx) += x(kVIdx) * ctheta;
  A(kPyIdx, kVIdx) += stheta;

  A(kThetaIdx, kPhiIdx) +=
      x(kVIdx) * time_step / (inter_axle_distance_ * cphi * cphi);
  A(kThetaIdx, kVIdx) += tphi * time_step / inter_axle_distance_;

  A(kKappaIdx, kPhiIdx) += 2.0 * time_step * u(kOmegaIdx) * tphi /
                           (cphi * cphi * inter_axle_distance_);

  A(kSIdx, kVIdx) += time_step;

  B(kPhiIdx, kOmegaIdx) = time_step;
  B(kVIdx, kAIdx) = time_step;
  B(kKappaIdx, kOmegaIdx) = time_step / (cphi * cphi * inter_axle_distance_);
}

inline float SinglePlayerCar7D::DistanceBetween(const VectorXf& x0,
                                                const VectorXf& x1) const {
  // Squared distance in position space.
  const float dx = x0(kPxIdx) - x1(kPxIdx);
  const float dy = x0(kPyIdx) - x1(kPyIdx);
  return dx * dx + dy * dy;
}

}  // namespace ilqgames

#endif
