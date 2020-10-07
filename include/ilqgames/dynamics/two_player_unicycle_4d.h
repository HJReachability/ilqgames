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
// Two player dynamics modeling a unicycle with velocity disturbance.
// State is [x, y, theta, v], u1 is [omega, a], u2 is [dx, dy] and dynamics are:
//                     \dot px    = v cos theta + dx
//                     \dot py    = v sin theta + dy
//                     \dot theta = omega
//                     \dot v     = a
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_DYNAMICS_TWO_PLAYER_UNICYCLE_4D_H
#define ILQGAMES_DYNAMICS_TWO_PLAYER_UNICYCLE_4D_H

#include <ilqgames/dynamics/multi_player_dynamical_system.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>

namespace ilqgames {

class TwoPlayerUnicycle4D : public MultiPlayerDynamicalSystem {
 public:
  ~TwoPlayerUnicycle4D() {}
  TwoPlayerUnicycle4D() : MultiPlayerDynamicalSystem(kNumXDims) {}

  // Compute time derivative of state.
  VectorXf Evaluate(Time t, const VectorXf& x,
                    const std::vector<VectorXf>& us) const;

  // Compute a discrete-time Jacobian linearization.
  LinearDynamicsApproximation Linearize(Time t, const VectorXf& x,
                                        const std::vector<VectorXf>& us) const;

  // Distance metric between two states.
  float DistanceBetween(const VectorXf& x0, const VectorXf& x1) const;

  // Position dimensions.
  std::vector<Dimension> PositionDimensions() const { return {kPxIdx, kPyIdx}; }

  // Getters.
  Dimension UDim(PlayerIndex player_idx) const {
    DCHECK(player_idx == 0 || player_idx == 1);
    return (player_idx == 0) ? kNumU1Dims : kNumU2Dims;
  }
  PlayerIndex NumPlayers() const { return kNumPlayers; }

  // Constexprs for state indices.
  static const Dimension kNumXDims;
  static const Dimension kPxIdx;
  static const Dimension kPyIdx;
  static const Dimension kThetaIdx;
  static const Dimension kVIdx;

  // Constexprs for control indices.
  static const PlayerIndex kNumPlayers;

  static const Dimension kNumU1Dims;
  static const Dimension kOmegaIdx;
  static const Dimension kAIdx;

  static const Dimension kNumU2Dims;
  static const Dimension kDxIdx;
  static const Dimension kDyIdx;
};  //\class TwoPlayerUnicycle4D

// ----------------------------- IMPLEMENTATION ----------------------------- //

inline VectorXf TwoPlayerUnicycle4D::Evaluate(
    Time t, const VectorXf& x, const std::vector<VectorXf>& us) const {
  CHECK_EQ(us.size(), NumPlayers());

  // Populate xdot one dimension at a time.
  VectorXf xdot(xdim_);
  xdot(kPxIdx) = x(kVIdx) * std::cos(x(kThetaIdx)) + us[1](kDxIdx);
  xdot(kPyIdx) = x(kVIdx) * std::sin(x(kThetaIdx)) + us[1](kDyIdx);
  xdot(kThetaIdx) = us[0](kOmegaIdx);
  xdot(kVIdx) = us[0](kAIdx);

  return xdot;
}

inline LinearDynamicsApproximation TwoPlayerUnicycle4D::Linearize(
    Time t, const VectorXf& x, const std::vector<VectorXf>& us) const {
  LinearDynamicsApproximation linearization(*this);

  const float ctheta = std::cos(x(kThetaIdx)) * time::kTimeStep;
  const float stheta = std::sin(x(kThetaIdx)) * time::kTimeStep;

  linearization.A(kPxIdx, kThetaIdx) += -x(kVIdx) * stheta;
  linearization.A(kPxIdx, kVIdx) += ctheta;

  linearization.A(kPyIdx, kThetaIdx) += x(kVIdx) * ctheta;
  linearization.A(kPyIdx, kVIdx) += stheta;

  linearization.Bs[0](kThetaIdx, kOmegaIdx) = time::kTimeStep;
  linearization.Bs[0](kVIdx, kAIdx) = time::kTimeStep;

  linearization.Bs[1](kPxIdx, kDxIdx) = time::kTimeStep;
  linearization.Bs[1](kPyIdx, kDyIdx) = time::kTimeStep;

  return linearization;
}

inline float TwoPlayerUnicycle4D::DistanceBetween(const VectorXf& x0,
                                                  const VectorXf& x1) const {
  // Squared distance in position space.
  const float dx = x0(kPxIdx) - x1(kPxIdx);
  const float dy = x0(kPyIdx) - x1(kPyIdx);
  return dx * dx + dy * dy;
}

}  // namespace ilqgames

#endif
