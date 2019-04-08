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

#include <ilqgames/dynamics/single_player_car_5d.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>

namespace ilqgames {

// Compute time derivative of state.
inline VectorXf SinglePlayerCar5D::Evaluate(Time t, const VectorXf& x,
                                            const VectorXf& u) const {
  VectorXf xdot(xdim_);
  xdot(kPxIdx) = x(kVIdx) * std::cos(x(kThetaIdx));
  xdot(kPyIdx) = x(kVIdx) * std::sin(x(kThetaIdx));
  xdot(kThetaIdx) = (x(kVIdx) / inter_axle_distance_) * std::tan(kPhiIdx);
  xdot(kPhiIdx) = u(kOmegaIdx);
  xdot(kVIdx) = u(kAIdx);

  return xdot;
}

// Compute a discrete-time Jacobian linearization.
void SinglePlayerCar5D::Linearize(Time t, const VectorXf& x, const VectorXf& u,
                                  Eigen::Ref<MatrixXf> A,
                                  Eigen::Ref<MatrixXf> B) const {
  const float ctheta = std::cos(x(kThetaIdx));
  const float stheta = std::sin(x(kThetaIdx));
  const float cphi = std::cos(x(kPhiIdx));
  const float tphi = std::tan(x(kPhiIdx));

  A(kPxIdx, kThetaIdx) = -x(kVIdx) * stheta;
  A(kPxIdx, kVIdx) = ctheta;

  A(kPyIdx, kThetaIdx) = x(kVIdx) * ctheta;
  A(kPyIdx, kVIdx) = stheta;

  A(kThetaIdx, kPhiIdx) = x(kVIdx) / (inter_axle_distance_ * cphi * cphi);
  A(kThetaIdx, kVIdx) = tphi / inter_axle_distance_;

  B(kPhiIdx, kOmegaIdx) = 1.0;
  B(kVIdx, kAIdx) = 1.0;
}

}  // namespace ilqgames
