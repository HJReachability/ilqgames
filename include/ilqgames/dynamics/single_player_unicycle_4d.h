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
// Single player dynamics modeling a unicycle. 4 states and 2 control inputs.
// State is [x, y, theta, v], control is [omega, a], and dynamics are:
//                     \dot px    = v cos theta
//                     \dot py    = v sin theta
//                     \dot theta = omega
//                     \dot v     = a
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_DYNAMICS_SINGLE_PLAYER_UNICYCLE_4D_H
#define ILQGAMES_DYNAMICS_SINGLE_PLAYER_UNICYCLE_4D_H

#include <ilqgames/dynamics/single_player_dynamical_system.h>
#include <ilqgames/utils/types.h>

namespace ilqgames {

class SinglePlayerUnicycle4D : public SinglePlayerDynamicalSystem {
 public:
  ~SinglePlayerUnicycle4D() {}
  SinglePlayerUnicycle4D()
      : SinglePlayerDynamicalSystem(kNumXDims, kNumUDims) {}

  // Compute time derivative of state.
  VectorXf Evaluate(Time t, const VectorXf& x, const VectorXf& u) const;

  // Compute a discrete-time Jacobian linearization.
  void Linearize(Time t, const VectorXf& x, const VectorXf& u,
                 Eigen::Ref<MatrixXf> A, Eigen::Ref<MatrixXf> B) const;

  // Constexprs for state indices.
  static constexpr Dimension kNumXDims = 4;
  static constexpr Dimension kPxIdx = 0;
  static constexpr Dimension kPyIdx = 1;
  static constexpr Dimension kThetaIdx = 2;
  static constexpr Dimension kVIdx = 3;

  // Constexprs for control indices.
  static constexpr Dimension kNumUDims = 2;
  static constexpr Dimension kOmegaIdx = 0;
  static constexpr Dimension kAIdx = 1;
};  //\class SinglePlayerUnicycle4D

// ----------------------------- IMPLEMENTATION ----------------------------- //

// Compute time derivative of state.
inline VectorXf SinglePlayerUnicycle4D::Evaluate(Time t, const VectorXf& x,
                                                 const VectorXf& u) const {
  VectorXf xdot(xdim_);
  xdot(kPxIdx) = x(kVIdx) * std::cos(x(kThetaIdx));
  xdot(kPyIdx) = x(kVIdx) * std::sin(x(kThetaIdx));
  xdot(kThetaIdx) = u(kOmegaIdx);
  xdot(kVIdx) = u(kAIdx);

  return xdot;
}

// Compute a discrete-time Jacobian linearization.
inline void SinglePlayerUnicycle4D::Linearize(Time t, const VectorXf& x,
                                              const VectorXf& u,
                                              Eigen::Ref<MatrixXf> A,
                                              Eigen::Ref<MatrixXf> B) const {
  const float ctheta = std::cos(x(kThetaIdx));
  const float stheta = std::sin(x(kThetaIdx));

  A(kPxIdx, kThetaIdx) = -x(kVIdx) * stheta;
  A(kPxIdx, kVIdx) = ctheta;

  A(kPyIdx, kThetaIdx) = x(kVIdx) * ctheta;
  A(kPyIdx, kVIdx) = stheta;

  B(kThetaIdx, kOmegaIdx) = 1.0;
  B(kVIdx, kAIdx) = 1.0;
}

}  // namespace ilqgames

#endif
