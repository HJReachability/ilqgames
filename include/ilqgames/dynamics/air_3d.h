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
// Air3D dynamics, from:
// https://www.cs.ubc.ca/~mitchell/Papers/publishedIEEEtac05.pdf.
//
// Here, two Dubins cars are navigating in relative coordinates, and the usual
// setup is a pursuit-evasion game.
//
// Dynamics are:
//                 \dot r_x = -v_e + v_p cos(r_theta) + u_e r_y
//                 \dot r_y = v_p sin(r_theta) - u_e r_x
//                 \dot r_theta = u_p - u_e
// and the convention below is that controls are "omega" and the evader is P1
// and the pursuer is P2.
//
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_DYNAMICS_AIR_3D_H
#define ILQGAMES_DYNAMICS_AIR_3D_H

#include <ilqgames/dynamics/multi_player_dynamical_system.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>

namespace ilqgames {

class Air3D : public MultiPlayerDynamicalSystem {
 public:
  ~Air3D() {}
  Air3D(float evader_speed, float pursuer_speed)
      : evader_speed_(evader_speed),
        pursuer_speed_(pursuer_speed),
        MultiPlayerDynamicalSystem(kNumXDims) {}

  // Compute time derivative of state.
  VectorXf Evaluate(Time t, const VectorXf& x,
                    const std::vector<VectorXf>& us) const;

  // Compute a discrete-time Jacobian linearization.
  LinearDynamicsApproximation Linearize(Time t, const VectorXf& x,
                                        const std::vector<VectorXf>& us) const;

  // Distance metric between two states.
  float DistanceBetween(const VectorXf& x0, const VectorXf& x1) const;

  // Getters.
  Dimension UDim(PlayerIndex player_idx) const {
    DCHECK(player_idx == 0 || player_idx == 1);
    return (player_idx == 0) ? kNumU1Dims : kNumU2Dims;
  }
  PlayerIndex NumPlayers() const { return kNumPlayers; }
  std::vector<Dimension> PositionDimensions() const { return {kRxIdx, kRyIdx}; }

  // Speed of each player.
  const float evader_speed_;
  const float pursuer_speed_;

  // Constexprs for state indices.
  static const Dimension kNumXDims;
  static const Dimension kRxIdx;
  static const Dimension kRyIdx;
  static const Dimension kRThetaIdx;

  // Constexprs for control indices.
  static const PlayerIndex kNumPlayers;

  static const Dimension kNumU1Dims;
  static const Dimension kOmega1Idx;

  static const Dimension kNumU2Dims;
  static const Dimension kOmega2Idx;
};  //\class TwoPlayerUnicycle4D

// ----------------------------- IMPLEMENTATION ----------------------------- //

inline VectorXf Air3D::Evaluate(Time t, const VectorXf& x,
                                const std::vector<VectorXf>& us) const {
  CHECK_EQ(us.size(), NumPlayers());

  // Populate xdot one dimension at a time.
  VectorXf xdot(xdim_);
  xdot(kRxIdx) = -evader_speed_ + pursuer_speed_ * std::cos(x(kRThetaIdx)) +
                 us[0](kOmega1Idx) * x(kRyIdx);
  xdot(kRyIdx) =
      pursuer_speed_ * std::sin(x(kRThetaIdx)) - us[0](kOmega1Idx) * x(kRxIdx);
  xdot(kRThetaIdx) = us[1](kOmega2Idx) - us[0](kOmega1Idx);

  return xdot;
}

inline LinearDynamicsApproximation Air3D::Linearize(
    Time t, const VectorXf& x, const std::vector<VectorXf>& us) const {
  LinearDynamicsApproximation linearization(*this);

  const float ctheta = std::cos(x(kRThetaIdx)) * time::kTimeStep;
  const float stheta = std::sin(x(kRThetaIdx)) * time::kTimeStep;

  linearization.A(kRxIdx, kRyIdx) += us[0](kOmega1Idx) * time::kTimeStep;
  linearization.A(kRxIdx, kRThetaIdx) -= pursuer_speed_ * stheta;

  linearization.A(kRyIdx, kRxIdx) -= us[0](kOmega1Idx) * time::kTimeStep;
  linearization.A(kRyIdx, kRThetaIdx) += pursuer_speed_ * ctheta;

  linearization.Bs[0](kRxIdx, kOmega1Idx) = x(kRyIdx) * time::kTimeStep;
  linearization.Bs[0](kRyIdx, kOmega1Idx) = -x(kRxIdx) * time::kTimeStep;
  linearization.Bs[0](kRThetaIdx, kOmega1Idx) = -time::kTimeStep;

  linearization.Bs[1](kRThetaIdx, kOmega2Idx) = time::kTimeStep;

  return linearization;
}

inline float Air3D::DistanceBetween(const VectorXf& x0,
                                    const VectorXf& x1) const {
  // Squared distance in position space.
  // NOTE: doesn't really make sense in this context, so logging an error.
  LOG(ERROR) << "Trying to compute distance between to relative states.";
  return (x0.head(2) - x1.head(2)).squaredNorm();
}

}  // namespace ilqgames

#endif
