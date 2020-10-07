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

#ifndef ILQGAMES_DYNAMICS_SINGLE_PLAYER_DUBINS_CAR_H
#define ILQGAMES_DYNAMICS_SINGLE_PLAYER_DUBINS_CAR_H

#include <ilqgames/dynamics/single_player_dynamical_system.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>

namespace ilqgames {

class SinglePlayerDubinsCar : public SinglePlayerDynamicalSystem {
 public:
  ~SinglePlayerDubinsCar() {}
  SinglePlayerDubinsCar(float v)
      : SinglePlayerDynamicalSystem(kNumXDims, kNumUDims), v_(v) {
    CHECK_GT(v_, 0.0);
  }

  // Compute time derivative of state.
  VectorXf Evaluate(Time t, const VectorXf& x, const VectorXf& u) const;

  // Compute a discrete-time Jacobian linearization.
  void Linearize(Time t, const VectorXf& x, const VectorXf& u,
                 Eigen::Ref<MatrixXf> A, Eigen::Ref<MatrixXf> B) const;

  // Position dimensions.
  std::vector<Dimension> PositionDimensions() const { return {kPxIdx, kPyIdx}; }

  // Constexprs for state indices.
  static const Dimension kNumXDims;
  static const Dimension kPxIdx;
  static const Dimension kPyIdx;
  static const Dimension kThetaIdx;

  // Constexprs for control indices.
  static const Dimension kNumUDims;
  static const Dimension kOmegaIdx;

 private:
  // Constant speed of the car.
  const float v_;
};  //\class SinglePlayerDubinsCar

// ----------------------------- IMPLEMENTATION ----------------------------- //

inline VectorXf SinglePlayerDubinsCar::Evaluate(Time t, const VectorXf& x,
                                                const VectorXf& u) const {
  VectorXf xdot(xdim_);
  xdot(kPxIdx) = v_ * std::cos(x(kThetaIdx));
  xdot(kPyIdx) = v_ * std::sin(x(kThetaIdx));
  xdot(kThetaIdx) = u(kOmegaIdx);

  return xdot;
}

inline void SinglePlayerDubinsCar::Linearize(Time t,
                                             const VectorXf& x,
                                             const VectorXf& u,
                                             Eigen::Ref<MatrixXf> A,
                                             Eigen::Ref<MatrixXf> B) const {
  const float ctheta = std::cos(x(kThetaIdx)) * time::kTimeStep;
  const float stheta = std::sin(x(kThetaIdx)) * time::kTimeStep;

  A(kPxIdx, kThetaIdx) += -v_ * stheta;
  A(kPyIdx, kThetaIdx) += v_ * ctheta;

  B(kThetaIdx, kOmegaIdx) = time::kTimeStep;
}

}  // namespace ilqgames

#endif
