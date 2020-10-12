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
// Single player dynamics modeling a 2D point mass. 4 states, 2 control inputs.
// State is [x, y, xdot, ydot], control is [ax, ay], and dynamics are:
//                     \dot px    = vx
//                     \dot py    = vy
//                     \dot vx    = ax
//                     \dot vy    = ay
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_DYNAMICS_SINGLE_PLAYER_POINT_MASS_2D_H
#define ILQGAMES_DYNAMICS_SINGLE_PLAYER_POINT_MASS_2D_H

#include <ilqgames/dynamics/single_player_dynamical_system.h>
#include <ilqgames/utils/types.h>

namespace ilqgames {

class SinglePlayerPointMass2D : public SinglePlayerDynamicalSystem {
 public:
  ~SinglePlayerPointMass2D() {}
  SinglePlayerPointMass2D()
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
  static const Dimension kVxIdx;
  static const Dimension kVyIdx;

  // Constexprs for control indices.
  static const Dimension kNumUDims;
  static const Dimension kAxIdx;
  static const Dimension kAyIdx;
};  //\class SinglePlayerPointMass2D

// ----------------------------- IMPLEMENTATION ----------------------------- //

inline VectorXf SinglePlayerPointMass2D::Evaluate(Time t, const VectorXf& x,
                                                  const VectorXf& u) const {
  VectorXf xdot(xdim_);
  xdot(kPxIdx) = x(kVxIdx);
  xdot(kPyIdx) = x(kVyIdx);
  xdot(kVxIdx) = u(kAxIdx);
  xdot(kVyIdx) = u(kAyIdx);

  return xdot;
}

inline void SinglePlayerPointMass2D::Linearize(Time t, const VectorXf& x,
                                               const VectorXf& u,
                                               Eigen::Ref<MatrixXf> A,
                                               Eigen::Ref<MatrixXf> B) const {
  A(kPxIdx, kVxIdx) += time::kTimeStep;
  A(kPyIdx, kVyIdx) += time::kTimeStep;

  B(kVxIdx, kAxIdx) = time::kTimeStep;
  B(kVyIdx, kAyIdx) = time::kTimeStep;
}

inline float SinglePlayerPointMass2D::DistanceBetween(
    const VectorXf& x0, const VectorXf& x1) const {
  // Squared distance in position space.
  const float dx = x0(kPxIdx) - x1(kPxIdx);
  const float dy = x0(kPyIdx) - x1(kPyIdx);
  return dx * dx + dy * dy;
}

}  // namespace ilqgames

#endif
