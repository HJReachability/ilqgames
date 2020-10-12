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
// Single player dynamics modeling a unicycle. 5 states and 2 control inputs.
// State is [x, y, theta, v], control is [omega, a], and dynamics are:
//                     \dot px    = v cos theta
//                     \dot py    = v sin theta
//                     \dot theta = omega
//                     \dot v     = a
//                     \dot s     = v
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_DYNAMICS_SINGLE_PLAYER_UNICYCLE_5D_H
#define ILQGAMES_DYNAMICS_SINGLE_PLAYER_UNICYCLE_5D_H

#include <ilqgames/dynamics/single_player_dynamical_system.h>
#include <ilqgames/utils/types.h>

namespace ilqgames {

class SinglePlayerUnicycle5D : public SinglePlayerDynamicalSystem {
 public:
  ~SinglePlayerUnicycle5D() {}
  SinglePlayerUnicycle5D()
      : SinglePlayerDynamicalSystem(kNumXDims, kNumUDims) {}

  // Compute time derivative of state.
  VectorXf Evaluate(Time t, const VectorXf& x, const VectorXf& u) const;

  // Compute a discrete-time Jacobian linearization.
  void Linearize(Time t, const VectorXf& x, const VectorXf& u,
                 Eigen::Ref<MatrixXf> A, Eigen::Ref<MatrixXf> B) const;

  // Distance metric between two states.
  float DistanceBetween(const VectorXf& x0, const VectorXf& x1) const;

  // Position dimensions.
  std::vector<Dimension> PositionDimensions() const { return {kPxIdx, kPyIdx}; }

  // Constexprs for state indices.
  static const Dimension kNumXDims;
  static const Dimension kPxIdx;
  static const Dimension kPyIdx;
  static const Dimension kThetaIdx;
  static const Dimension kVIdx;
  static const Dimension kSIdx;

  // Constexprs for control indices.
  static const Dimension kNumUDims;
  static const Dimension kOmegaIdx;
  static const Dimension kAIdx;
};  //\class SinglePlayerUnicycle5D

// ----------------------------- IMPLEMENTATION ----------------------------- //

inline VectorXf SinglePlayerUnicycle5D::Evaluate(Time t, const VectorXf& x,
                                                 const VectorXf& u) const {
  VectorXf xdot(xdim_);
  xdot(kPxIdx) = x(kVIdx) * std::cos(x(kThetaIdx));
  xdot(kPyIdx) = x(kVIdx) * std::sin(x(kThetaIdx));
  xdot(kThetaIdx) = u(kOmegaIdx);
  xdot(kVIdx) = u(kAIdx);
  xdot(kSIdx) = x(kVIdx);

  return xdot;
}

inline void SinglePlayerUnicycle5D::Linearize(Time t, const VectorXf& x,
                                              const VectorXf& u,
                                              Eigen::Ref<MatrixXf> A,
                                              Eigen::Ref<MatrixXf> B) const {
  const float ctheta = std::cos(x(kThetaIdx)) * time::kTimeStep;
  const float stheta = std::sin(x(kThetaIdx)) * time::kTimeStep;

  A(kPxIdx, kThetaIdx) += -x(kVIdx) * stheta;
  A(kPxIdx, kVIdx) += ctheta;

  A(kPyIdx, kThetaIdx) += x(kVIdx) * ctheta;
  A(kPyIdx, kVIdx) += stheta;

  A(kSIdx, kVIdx) += time::kTimeStep;

  B(kThetaIdx, kOmegaIdx) = time::kTimeStep;
  B(kVIdx, kAIdx) = time::kTimeStep;
}

inline float SinglePlayerUnicycle5D::DistanceBetween(const VectorXf& x0,
                                                     const VectorXf& x1) const {
  // Squared distance in position space.
  const float dx = x0(kPxIdx) - x1(kPxIdx);
  const float dy = x0(kPyIdx) - x1(kPyIdx);
  return dx * dx + dy * dy;
}

}  // namespace ilqgames

#endif
