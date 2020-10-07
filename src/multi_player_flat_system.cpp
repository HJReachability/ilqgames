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
// Base class for all multi-player flat systems. Supports (discrete-time)
// linearization and integration.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/dynamics/multi_player_flat_system.h>
#include <ilqgames/utils/linear_dynamics_approximation.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>

namespace ilqgames {

VectorXf MultiPlayerFlatSystem::Integrate(
    Time time_interval, const VectorXf& xi0,
    const std::vector<VectorXf>& vs) const {
  // Number of integration steps and corresponding time step.
  constexpr size_t kNumIntegrationSteps = 2;
  const double dt = time::kTimeStep / static_cast<Time>(kNumIntegrationSteps);

  CHECK_NOTNULL(continuous_linear_system_.get());
  auto xi_dot = [this, &vs](const VectorXf& xi) {
    VectorXf deriv = this->continuous_linear_system_->A * xi;
    for (size_t ii = 0; ii < NumPlayers(); ii++)
      deriv += this->continuous_linear_system_->Bs[ii] * vs[ii];

    return deriv;
  };  // xi_dot

  // RK4 integration. See https://en.wikipedia.org/wiki/Runge-Kutta_methods for
  // further details.
  VectorXf xi(xi0);
  for (Time t = 0.0; t < time_interval - 0.5 * dt; t += dt) {
    const VectorXf k1 = dt * xi_dot(xi);
    const VectorXf k2 = dt * xi_dot(xi + 0.5 * k1);
    const VectorXf k3 = dt * xi_dot(xi + 0.5 * k2);
    const VectorXf k4 = dt * xi_dot(xi + k3);

    xi += (k1 + 2.0 * (k2 + k3) + k4) / 6.0;
  }

  return xi;
}

}  // namespace ilqgames
